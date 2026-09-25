#!/usr/bin/env python
"""synth-intents v3: coding_ci world. 编码 agent 或开发者处理代码与 CI 时眼前的 state, 题目问这时要做的判断.

  python datasets/synth-intents-v3/coding_ci.py [--out DIR]

7 种 state, 每种 3 份, 每种配 3-5 道题, 每道题问这一种的全部 3 份:
  ci  一次流水线运行的日志 (log)          tr  失败测试的输出与堆栈, 加测试数据 (log)
  df  一个 pull request 的 diff (structured)  dp  依赖安装失败时的 JSON (json)
  mx  跨平台测试矩阵 (table)              hs  带 CI 结果的 git 历史 (log)
  rv  一次 code review 的评论串 (dialogue)
选项在 3 份之间通用的题写成一道 (ask_all), 3 份的答案互不相同; 选项是某一份里的具体东西的题每份各一道
(ask_each), 问法一字不差. 没有自然顺序的选项用 random.Random(题 id) 打乱. 答案由 state 定死的写进 stated;
要判断的题 (每种最多一道) 不写. 键的意思与写作规则见 schema.py.
"""

import random

from schema import YES_NO, Question, Text, main

DOMAIN = "coding_ci"
LABEL = "Code and CI state"

QUESTIONS: list[Question] = []
TEXTS: list[Text] = []
_TEXT: dict[str, Text] = {}


def state(suffix: str, style: str, text) -> None:
    t = Text(id=f"{DOMAIN}_{suffix}", style=style, text=text)
    TEXTS.append(t)
    _TEXT[suffix] = t


def ask_all(qid: str, ask, options: list[str], answers: dict, *, keep: bool = False, needs=(),
            judged: bool = True) -> None:
    """一道题问几份 state. answers: {后缀: 答案原文 或 None (不写明)}. keep=True 保持选项的自然顺序."""
    opts = list(options)
    if not keep and opts != YES_NO:
        random.Random(qid).shuffle(opts)
    QUESTIONS.append(Question(id=qid, ask=[ask] if isinstance(ask, str) else list(ask), options=opts,
                              needs=list(needs), texts=[_TEXT[s].id for s in answers], judged=judged))
    for s, a in answers.items():
        if a is not None:
            _TEXT[s].stated[qid] = opts.index(a)


def ask_each(slug: str, ask, per: dict, *, keep: bool = False, needs=(), judged: bool = True) -> None:
    """每份 state 各一道题 (id 为 <后缀>_<slug>), 问法一字不差, 选项取自那一份. per: {后缀: (选项, 答案原文)}."""
    for s, (options, answer) in per.items():
        ask_all(f"{s}_{slug}", ask, options, {s: answer}, keep=keep, needs=needs, judged=judged)


def lines(*rows: str) -> str:
    return "\n".join(rows)


# ---------------------------------------------------------------- ci: 一次流水线运行的日志
STAGES = ["checkout", "install dependencies", "lint", "typecheck", "build", "unit tests", "integration tests",
          "package", "deploy to staging"]
SKIP = "skipped: an earlier stage failed"

state("ci1", "log", lines(
    "Pipeline run 5521 | repo shop-web | branch feature/export-csv | trigger: pull request | runner: linux-medium",
    "09:14:02 [checkout] fetching feature/export-csv, depth 50",
    "09:14:06 [checkout] done in 4s",
    "09:14:07 [install dependencies] attempt 1 of 3: npm ci",
    "09:15:07 [install dependencies] npm ERR! network request to the package registry timed out after 60s",
    "09:15:08 [install dependencies] attempt 2 of 3: npm ci",
    "09:15:49 [install dependencies] added 812 packages in 41s",
    "09:15:49 [install dependencies] done",
    "09:15:50 [lint] eslint src --max-warnings 20",
    "09:16:02 [lint] 0 errors, 7 warnings",
    "09:16:02 [lint] done in 12s",
    "09:16:03 [typecheck] tsc --noEmit",
    "09:16:21 [typecheck] src/export/csv.ts(41,18): error TS2345: Argument of type 'string' is not assignable "
    "to parameter of type 'number'.",
    "09:16:21 [typecheck] Found 1 error in src/export/csv.ts",
    "09:16:21 [typecheck] FAILED, exit code 2",
    f"09:16:22 [build] {SKIP}",
    f"09:16:22 [unit tests] {SKIP}",
    f"09:16:22 [integration tests] {SKIP}",
    f"09:16:22 [package] {SKIP}",
    f"09:16:22 [deploy to staging] {SKIP}",
    "09:16:23 pipeline finished: failed",
))

state("ci2", "log", lines(
    "Pipeline run 7310 | repo stock-api | branch main | trigger: push | runner: linux-large",
    "13:40:11 [checkout] fetching main, depth 1",
    "13:40:13 [checkout] done in 2s",
    "13:40:14 [install dependencies] attempt 1 of 3: pip install -r requirements.txt",
    "13:41:02 [install dependencies] Successfully installed 64 packages",
    "13:41:02 [install dependencies] done",
    "13:41:03 [lint] ruff check .",
    "13:41:05 [lint] app/models.py:88:5: E722 Do not use bare except",
    "13:41:05 [lint] Found 1 error.",
    "13:41:05 [lint] FAILED, exit code 1 (this stage is allowed to fail, continuing)",
    "13:41:06 [typecheck] mypy app",
    "13:41:30 [typecheck] Success: no issues found in 41 source files",
    "13:41:31 [build] python -m build --wheel",
    "13:41:44 [build] done in 13s",
    "13:41:45 [unit tests] pytest tests/unit -q",
    "13:42:31 [unit tests] 318 passed in 46s",
    "13:42:32 [integration tests] starting service containers: postgres 15, redis 7",
    "13:42:50 [integration tests] pytest tests/integration -q",
    "13:43:20 [integration tests] E   psycopg.OperationalError: connection to server at 10.0.4.12, port 5432 "
    "failed: Connection refused",
    "13:43:20 [integration tests] 22 errors in 30s",
    "13:43:21 [integration tests] FAILED, exit code 1",
    f"13:43:21 [package] {SKIP}",
    f"13:43:21 [deploy to staging] {SKIP}",
    "13:43:22 pipeline finished: failed",
))

state("ci3", "log", lines(
    "Pipeline run 6408 | repo image-worker | branch release/2.4 | trigger: nightly schedule | runner: linux-small",
    "02:00:03 [checkout] fetching release/2.4, depth 1",
    "02:00:06 [checkout] done in 3s",
    "02:00:07 [install dependencies] attempt 1 of 3: go mod download",
    "02:01:07 [install dependencies] dial tcp 10.0.9.3:443: i/o timeout while fetching modules",
    "02:01:17 [install dependencies] attempt 2 of 3: go mod download",
    "02:02:17 [install dependencies] dial tcp 10.0.9.3:443: i/o timeout while fetching modules",
    "02:02:37 [install dependencies] attempt 3 of 3: go mod download",
    "02:03:37 [install dependencies] dial tcp 10.0.9.3:443: i/o timeout while fetching modules",
    "02:03:37 [install dependencies] FAILED, exit code 1, no attempts left",
    f"02:03:38 [lint] {SKIP}",
    f"02:03:38 [typecheck] {SKIP}",
    f"02:03:38 [build] {SKIP}",
    f"02:03:38 [unit tests] {SKIP}",
    f"02:03:38 [integration tests] {SKIP}",
    f"02:03:38 [package] {SKIP}",
    f"02:03:38 [deploy to staging] {SKIP}",
    "02:03:39 [post] upload runner logs: write error, no space left on device (ignored)",
    "02:03:40 pipeline finished: failed",
))

ask_all("ci_stop", ["Which stage made the pipeline stop?", "The run ended early. Which stage's failure ended it?"],
        STAGES, {"ci1": "typecheck", "ci2": "integration tests", "ci3": "install dependencies"}, keep=True)
ask_all("ci_attempts", ["How many attempts did the install dependencies stage make?",
                        "How many times did the pipeline try to install dependencies?"],
        ["one", "two", "three"], {"ci1": "two", "ci2": "one", "ci3": "three"}, keep=True)
ask_all("ci_unit_ran", ["Did the unit tests stage run in this pipeline?", "Were the unit tests run?"],
        YES_NO, {"ci1": "no", "ci2": "yes", "ci3": "no"})
ask_all("ci_cause", ["What made the stage that stopped the pipeline fail?", "Why did the pipeline stop?"],
        ["Downloading dependencies timed out", "A value of the wrong type was passed to a function",
         "The tests could not reach the database", "The runner's disk was full", "The code broke a lint rule"],
        {"ci1": "A value of the wrong type was passed to a function",
         "ci2": "The tests could not reach the database",
         "ci3": "Downloading dependencies timed out"})
ask_all("ci_next", ["What should the developer do first to get this pipeline passing?",
                    "What is the best next step for this pipeline?"],
        ["Fix the code that the failing stage reports", "Re-run the pipeline later without changing anything",
         "Check why the database container refused connections", "Free up disk space on the runner",
         "Turn off the lint stage"],
        {"ci1": None, "ci2": None, "ci3": None})

# ---------------------------------------------------------------- tr: 失败测试的输出与堆栈, 加测试数据
PY_SITE = ".venv/lib/python3.12/site-packages"
state("tr1", "log", lines(
    "$ pytest tests/reports -q",
    "tests/reports/test_charts.py ....                                        [ 28%]",
    "tests/reports/test_export.py ......                                      [ 71%]",
    "tests/reports/test_totals.py ..F.                                        [100%]",
    "",
    "=================================== FAILURES ===================================",
    "___________________________ test_totals_skip_refunds ___________________________",
    "",
    "    def test_totals_skip_refunds(client):",
    "        rows = [",
    '            {"id": 3, "region": "north", "total": 12.0},',
    '            {"id": 5, "region": "south", "total": 7.5},',
    '            {"id": 7, "country": "PT", "total": 18.5},',
    '            {"id": 9, "region": "north", "total": -4.0, "refund": True},',
    "        ]",
    '>       resp = client.post("/reports/totals", json=rows)',
    "",
    "tests/reports/test_totals.py:48:",
    "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _",
    f"{PY_SITE}/flask/testing.py:245: in post",
    "    return self.open(*args, **kw)",
    f"{PY_SITE}/werkzeug/test.py:1161: in open",
    "    response = self.run_wsgi_app(request.environ, buffered=buffered)",
    f"{PY_SITE}/flask/app.py:1498: in wsgi_app",
    "    response = self.full_dispatch_request()",
    "app/views/reports.py:22: in post_totals",
    "    return jsonify(totals_by_region(request.get_json()))",
    "app/reports/totals.py:14: in totals_by_region",
    "    name = region_of(row)",
    "app/reports/regions.py:6: in region_of",
    '    return row["region"].strip().lower()',
    "E   KeyError: 'region'",
    "=========================== short test summary info ============================",
    "FAILED tests/reports/test_totals.py::test_totals_skip_refunds - KeyError: 'region'",
    "========================= 1 failed, 13 passed in 0.84s =========================",
))

REACT_DOM = "node_modules/react-dom/cjs/react-dom.development.js"
state("tr2", "log", lines(
    "$ npx jest src/components",
    " PASS  src/components/__tests__/Header.test.jsx",
    " FAIL  src/components/__tests__/ProjectTable.test.jsx",
    "  ProjectTable",
    "    ✓ renders a header row (31 ms)",
    "    ✕ shows the owner of every project (18 ms)",
    "    ✓ sorts by name when the header is clicked (22 ms)",
    "",
    "  ● ProjectTable › shows the owner of every project",
    "",
    "    TypeError: Cannot read properties of undefined (reading 'name')",
    "",
    "      10 | export function ownerLabel(project) {",
    "      11 |   const initials = project.code.slice(0, 2);",
    "    > 12 |   return `${project.owner.name} (${initials})`;",
    "         |                           ^",
    "      13 | }",
    "",
    "      at ownerLabel (src/utils/owner.js:12:27)",
    "      at src/components/ProjectTable.jsx:40:18",
    "          at Array.map (<anonymous>)",
    "      at ProjectTable (src/components/ProjectTable.jsx:38:22)",
    f"      at renderWithHooks ({REACT_DOM}:16305:18)",
    f"      at mountIndeterminateComponent ({REACT_DOM}:20074:13)",
    "      at renderRoot (node_modules/testing-kit/dist/render.js:88:10)",
    "      at Object.<anonymous> (src/components/__tests__/ProjectTable.test.jsx:21:5)",
    "",
    "Tests: 1 failed, 5 passed, 6 total",
    "",
    "$ grep -n projects src/components/__tests__/ProjectTable.test.jsx",
    '3:import projects from "../../fixtures/projects.json";',
    "21:    render(<ProjectTable projects={projects} />);",
    "",
    "$ cat src/fixtures/projects.json",
    "[",
    '  {"code": "AT-1", "name": "Atlas", "owner": {"name": "Kim"}},',
    '  {"code": "BI-2", "name": "Birch", "lead": {"name": "Ola"}},',
    '  {"code": "CE-3", "name": "Cedar", "owner": {"name": "Rui"}},',
    '  {"code": "DU-4", "name": "Dune", "owner": {"name": ""}}',
    "]",
))

state("tr3", "log", lines(
    "$ ./gradlew test --tests LayoutBuilderTest",
    "> Task :compileJava UP-TO-DATE",
    "> Task :compileTestJava",
    "> Task :test",
    "",
    "LayoutBuilderTest > buildsEmptyLayout() PASSED",
    "",
    "LayoutBuilderTest > buildsGridFromTheme() FAILED",
    '    java.lang.NumberFormatException: For input string: "12px"',
    "        at java.base/java.lang.NumberFormatException.forInputString(NumberFormatException.java:67)",
    "        at java.base/java.lang.Integer.parseInt(Integer.java:662)",
    "        at java.base/java.lang.Integer.parseInt(Integer.java:778)",
    "        at com.example.layout.SizeParser.parse(SizeParser.java:27)",
    "        at com.example.layout.Spacing.fromTheme(Spacing.java:41)",
    "        at com.example.layout.LayoutBuilder.build(LayoutBuilder.java:63)",
    "        at com.example.layout.LayoutBuilderTest.buildsGridFromTheme(LayoutBuilderTest.java:35)",
    "        at java.base/jdk.internal.reflect.DirectMethodHandleAccessor.invoke(DirectMethodHandleAccessor.java:103)",
    "        at java.base/java.lang.reflect.Method.invoke(Method.java:580)",
    "        at org.junit.platform.commons.util.ReflectionUtils.invokeMethod(ReflectionUtils.java:728)",
    "        at org.junit.jupiter.engine.execution.MethodInvocation.proceed(MethodInvocation.java:60)",
    "",
    "LayoutBuilderTest > buildsRowsInOrder() PASSED",
    "",
    "3 tests completed, 1 failed",
    "",
    "$ cat src/test/resources/theme.properties",
    "spacing.small=4",
    "spacing.medium=8",
    "spacing.gutter=16",
    "spacing.large=12px",
))

ask_each("file", "Which of the project's own files is closest to the point where the error was raised?", {
    "tr1": (["tests/reports/test_charts.py", "tests/reports/test_export.py", "tests/reports/test_totals.py",
             f"{PY_SITE}/flask/testing.py", f"{PY_SITE}/werkzeug/test.py", f"{PY_SITE}/flask/app.py",
             "app/views/reports.py", "app/reports/totals.py", "app/reports/regions.py"],
            "app/reports/regions.py"),
    "tr2": (["src/components/__tests__/Header.test.jsx", "src/components/__tests__/ProjectTable.test.jsx",
             "src/utils/owner.js", "src/components/ProjectTable.jsx", REACT_DOM,
             "node_modules/testing-kit/dist/render.js", "src/fixtures/projects.json"],
            "src/utils/owner.js"),
    "tr3": (["NumberFormatException.java", "Integer.java", "SizeParser.java", "Spacing.java", "LayoutBuilder.java",
             "LayoutBuilderTest.java", "DirectMethodHandleAccessor.java", "Method.java", "ReflectionUtils.java",
             "MethodInvocation.java", "src/test/resources/theme.properties"],
            "SizeParser.java"),
})
ask_all("tr_kind", ["What kind of mistake does the error point to?",
                    "Which of these describes the error that was raised?"],
        ["A property was read from a value that was undefined or null",
         "A dictionary was asked for a key it does not contain",
         "A string could not be parsed as a whole number",
         "A result differed from the value the test expected",
         "A list was indexed past its last element"],
        {"tr1": "A dictionary was asked for a key it does not contain",
         "tr2": "A property was read from a value that was undefined or null",
         "tr3": "A string could not be parsed as a whole number"})
ask_each("input", "Which piece of the test data caused the error?", {
    "tr1": (["The row with id 3", "The row with id 5", "The row with id 7", "The row with id 9"],
            "The row with id 7"),
    "tr2": (["The project Atlas", "The project Cedar", "The project Birch", "The project Dune"],
            "The project Birch"),
    "tr3": (["The spacing.small entry", "The spacing.large entry", "The spacing.medium entry",
             "The spacing.gutter entry"], "The spacing.large entry"),
})
ask_all("tr_own", ["Was the exception raised inside the project's own code?",
                   "Did the error come from the project's own code rather than from a library?"],
        YES_NO, {"tr1": "yes", "tr2": "yes", "tr3": "no"})

# ---------------------------------------------------------------- df: 一个 pull request 的 diff
# unified diff 去掉了 hunk 头 (含 @), 文件按路径排序 (git 的顺序). 每份 4 个函数里只有一个改了逻辑,
# 其余只改名 / 加类型 / 加注释; 与标题无关的文件在 3 份里分别排第一、最后、中间.
state("df1", "structured", lines(
    "Pull request 318: Tidy up the size helpers",
    "Files changed: deploy/nginx.conf, src/sizes.py, tests/test_sizes.py",
    "",
    "--- a/deploy/nginx.conf",
    "+++ b/deploy/nginx.conf",
    " events {",
    "-    worker_connections 512;",
    "+    worker_connections 1024;",
    " }",
    "--- a/src/sizes.py",
    "+++ b/src/sizes.py",
    "-def parse_count(text):",
    "+def parse_count(text: str) -> int:",
    '     return int(text.replace(",", ""))',
    " ",
    " ",
    "-def clamp(value, low, high):",
    "-    return max(low, min(value, high))",
    "+def clamp(n, low, high):",
    "+    return max(low, min(n, high))",
    " ",
    " ",
    "-def size_label(count):",
    "-    if count > 100:",
    "+def size_label(count: int) -> str:",
    "+    if count >= 100:",
    '         return "large"',
    "     if count > 10:",
    '         return "medium"',
    '     return "small"',
    " ",
    " ",
    " def human_bytes(n):",
    "+    \"\"\"Format a byte count, for example 2048 becomes '2.0 KB'.\"\"\"",
    '     for unit in ["B", "KB", "MB"]:',
    "         if n < 1024:",
    '             return f"{n:.1f} {unit}"',
    "         n /= 1024",
    '     return f"{n:.1f} GB"',
    "--- a/tests/test_sizes.py",
    "+++ b/tests/test_sizes.py",
    " def test_size_label():",
    '     assert size_label(5) == "small"',
    '+    assert size_label(50) == "medium"',
))

state("df2", "structured", lines(
    "Pull request 1204: Rename the helpers in the date utils",
    "Files changed: src/components/DatePicker.jsx, src/dates.js, src/utils/strings.js",
    "",
    "--- a/src/components/DatePicker.jsx",
    "+++ b/src/components/DatePicker.jsx",
    '-import { daysBetween, relativeLabel } from "../dates";',
    '+import { differenceInDays, relativeLabel } from "../dates";',
    "-  const label = relativeLabel(daysBetween(today, value));",
    "+  const label = relativeLabel(differenceInDays(today, value));",
    "--- a/src/dates.js",
    "+++ b/src/dates.js",
    "-export function daysBetween(a, b) {",
    "-  return Math.round((b - a) / DAY_MS);",
    "+export function differenceInDays(start, end) {",
    "+  return Math.round((end - start) / DAY_MS);",
    " }",
    " ",
    " export function relativeLabel(days) {",
    '-  if (days < 7) return "this week";',
    '+  if (days <= 7) return "this week";',
    '   if (days < 31) return "this month";',
    '   return "earlier";',
    " }",
    " ",
    "-export function isWeekend(d) {",
    "-  return d.getDay() === 0 || d.getDay() === 6;",
    "+export function isWeekend(date) {",
    "+  const day = date.getDay();",
    "+  return day === 0 || day === 6;",
    " }",
    " ",
    "+// Monday is the first day of the week here.",
    " export function startOfWeek(date) {",
    "   const back = (date.getDay() + 6) % 7;",
    "   return addDays(date, -back);",
    " }",
    "--- a/src/utils/strings.js",
    "+++ b/src/utils/strings.js",
    "-// Pads a string on the left untill it reaches the given width.",
    "+// Pads a string on the left until it reaches the given width.",
    " export function padLeft(text, width) {",
))

state("df3", "structured", lines(
    "Pull request 77: Retry uploads when the server is rate limiting",
    "Files changed: docs/upload.md, internal/log/format.go, internal/upload/retry.go",
    "",
    "--- a/docs/upload.md",
    "+++ b/docs/upload.md",
    "-Uploads are retried when the server answers with a 5xx status.",
    "+Uploads are retried when the server answers with a 5xx status or with 429 Too Many Requests.",
    "--- a/internal/log/format.go",
    "+++ b/internal/log/format.go",
    "-// formatLogLine renders one log entry as key=value pairs seperated by spaces.",
    "+// formatLogLine renders one log entry as key=value pairs separated by spaces.",
    " func formatLogLine(e Entry) string {",
    "--- a/internal/upload/retry.go",
    "+++ b/internal/upload/retry.go",
    " const maxAttempts = 4",
    " ",
    " func shouldRetry(status int) bool {",
    "-\treturn status >= 500",
    "+\treturn status >= 500 || status == 429",
    " }",
    " ",
    "-func backoff(attempt int) time.Duration {",
    "-\treturn time.Duration(attempt) * time.Second",
    "+func backoff(n int) time.Duration {",
    "+\treturn time.Duration(n) * time.Second",
    " }",
    " ",
    "-func jitter(d time.Duration) time.Duration { return d + d/10 }",
    "+// jitter adds ten percent so that clients do not retry in step.",
    "+func jitter(d time.Duration) time.Duration {",
    "+\treturn d + d/10",
    "+}",
))

ask_each("logic", "Which function does this diff change the logic of, rather than just its names, types or comments?", {
    "df1": (["parse_count", "size_label", "clamp", "human_bytes"], "size_label"),
    "df2": (["differenceInDays", "relativeLabel", "isWeekend", "startOfWeek"], "relativeLabel"),
    "df3": (["shouldRetry", "backoff", "jitter", "formatLogLine"], "shouldRetry"),
})
ask_each("call", "After this diff, which of these calls returns something different from before?", {
    "df1": (["size_label(100)", "size_label(99)", "clamp(5, 0, 10)", "human_bytes(2048)"], "size_label(100)"),
    "df2": (["relativeLabel(7)", "relativeLabel(6)", "relativeLabel(31)", "startOfWeek(today)"], "relativeLabel(7)"),
    "df3": (["shouldRetry(429)", "shouldRetry(500)", "shouldRetry(404)", "backoff(4)"], "shouldRetry(429)"),
})
ask_each("unrelated", "Which changed file has nothing to do with the pull request's title?", {
    "df1": (["src/sizes.py", "tests/test_sizes.py", "deploy/nginx.conf"], "deploy/nginx.conf"),
    "df2": (["src/utils/strings.js", "src/dates.js", "src/components/DatePicker.jsx"], "src/utils/strings.js"),
    "df3": (["internal/upload/retry.go", "docs/upload.md", "internal/log/format.go"], "internal/log/format.go"),
})
ask_all("df_tests", ["Does this diff change any test file?", "Are any tests touched by this pull request?"],
        YES_NO, {"df1": "yes", "df2": "no", "df3": "no"})

# ---------------------------------------------------------------- dp: 依赖安装失败时的 JSON
# lockfile 的填充行用代码拼 (名字取自两张词表的组合, 与手写的包名不重), 只有手写的那个包出现两次.
PKG_A = ["ansi", "arg", "async", "brace", "buffer", "cache", "color", "cookie", "cross", "deep", "dot", "env",
         "event", "fast", "file", "flag", "gzip", "ini", "json", "lru", "mime", "range", "safe", "source",
         "string", "strip", "type", "uri", "wrap", "yaml", "char", "text"]
PKG_B = ["tint", "walk", "merge", "parse", "map", "guard", "split", "pick", "trim", "lock", "queue", "store",
         "shim", "list", "cast", "fold", "pipe", "tree"]


def lockfile(seed: str, fixed: list[str], n: int) -> list[str]:
    """n 行 "名字 版本", 按名字排序: fixed 原样放进去, 其余用不与 fixed 重名的填充包补齐."""
    rng = random.Random(seed)
    taken = {row.split()[0] for row in fixed}
    names = [f"{a}-{b}" for a in PKG_A for b in PKG_B if f"{a}-{b}" not in taken]
    rng.shuffle(names)
    filler = [f"{x} {rng.randint(0, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 12)}" for x in names[:n - len(fixed)]]
    return sorted(fixed + filler, key=lambda r: (r.split()[0], r))


def package_names(rows: list[str]) -> list[str]:
    return sorted({r.split()[0] for r in rows})


DP1_LOCK = lockfile("dp1", ["chart-kit 4.2.0", "view-core 17.0.2", "form-lite 2.3.4", "http-lite 1.6.2",
                            "date-kit 3.0.1", "icon-set 5.1.0", "glob-walk 7.2.3", "glob-walk 10.3.10",
                            "glob-walk-sync 1.0.4"], 38)
state("dp1", "json", {
    "command": "npm install",
    "project": "shop-web 1.0.0",
    "requires": {"chart-kit": "^4.2.0", "view-core": "^17.0.2", "form-lite": "^2.3.0", "http-lite": "^1.6.0",
                 "date-kit": "^3.0.1", "icon-set": "^5.1.0"},
    "lockfile": DP1_LOCK,
    "error": "npm ERR! ERESOLVE unable to resolve dependency tree. While resolving: shop-web 1.0.0. "
             "Found: view-core 17.0.2, from the root project. Could not resolve dependency: "
             "peer view-core \"^18.0.0\" from chart-kit 4.2.0, from the root project.",
})

DP2_LOCK = lockfile("dp2", ["net-client 2.1.0", "sign-kit 0.9.1", "tls-lite 0.21.4", "hash-core 0.16.20",
                            "serde-lite 1.0.197", "log-kit 0.4.21", "cli-args 4.5.2", "byte-pool 1.5.0",
                            "byte-pool 0.4.12"], 36)
state("dp2", "json", {
    "command": "cargo build",
    "project": "edge-proxy 0.3.0",
    "requires": {"net-client": "2.1", "tls-lite": "0.21", "serde-lite": "1.0", "log-kit": "0.4",
                 "cli-args": "4.5", "byte-pool": "1.5"},
    "lockfile": DP2_LOCK,
    "error": "error: failed to select a version for `hash-core`.\n"
             "    ... required by package `sign-kit v0.9.1`\n"
             "    ... which satisfies dependency `sign-kit = \"^0.9\"` of package `net-client v2.1.0`\n"
             "versions that meet the requirements `^0.17` are: 0.17.8, 0.17.7\n"
             "all possible versions conflict with previously selected packages.\n"
             "  previously selected package `hash-core v0.16.20`\n"
             "    ... which satisfies dependency `hash-core = \"^0.16\"` of package `tls-lite v0.21.4`",
})

DP3_LOCK = lockfile("dp3", ["form-kit 3.1.0", "form-kit-zod 1.2.0", "schema-lib 3.20.2", "router-x 6.2.1",
                            "store-lite 4.4.1", "date-range-picker 2.0.3", "path-scan 1.0.2", "path-scan 2.3.0"], 37)
state("dp3", "json", {
    "command": "pnpm install --frozen-lockfile",
    "project": "admin-ui 2.0.0",
    "requires": {"form-kit": "^3.1.0", "form-kit-zod": "^1.2.0", "router-x": "^6.2.0", "store-lite": "^4.4.1",
                 "date-range-picker": "^2.0.0"},
    "lockfile": DP3_LOCK,
    "error": "ERR_PNPM_PEER_DEP_ISSUES  Unmet peer dependencies\n"
             ".\n"
             "├─┬ form-kit-zod 1.2.0\n"
             "│ └── ✕ unmet peer schema-lib ^3.22.0: found 3.20.2 in form-kit\n"
             "└─┬ form-kit 3.1.0\n"
             "  └── schema-lib 3.20.2",
})

ask_each("twice", "Which package appears in `lockfile` at two different versions?", {
    "dp1": (package_names(DP1_LOCK), "glob-walk"),
    "dp2": (package_names(DP2_LOCK), "byte-pool"),
    "dp3": (package_names(DP3_LOCK), "path-scan"),
}, needs=["lockfile"], judged=False)
ask_each("asker", "According to `error`, which package asks for a version that cannot be met?", {
    "dp1": (["chart-kit", "view-core", "form-lite", "date-kit", "icon-set", "shop-web"], "chart-kit"),
    "dp2": (["hash-core", "net-client", "tls-lite", "sign-kit", "edge-proxy", "log-kit"], "sign-kit"),
    "dp3": (["form-kit", "form-kit-zod", "schema-lib", "router-x", "store-lite", "date-range-picker"],
            "form-kit-zod"),
}, needs=["error"])
ask_all("dp_direct", ["`error` names a package whose version is too old for another package. "
                      "Is that package listed in `requires`?",
                      "Is the package that `error` finds at too old a version one of the entries in `requires`?"],
        YES_NO, {"dp1": "yes", "dp2": "no", "dp3": "no"}, needs=["error", "requires"])

# ---------------------------------------------------------------- mx: 跨平台 × Python 版本的测试矩阵
# 每份 3 个 job 两次都失败、1 个 job 首次失败重跑通过. mx2 的行按 Python 版本排.
MX_HEAD = "job | os | python | first try | re-run | failing test"
PASS = "pass | - | -"
state("mx1", "table", lines(
    "Test matrix | run 4471 | branch feature/tz-aware | started 25 Sep 10:12",
    MX_HEAD,
    f"1 | ubuntu | 3.10 | {PASS}",
    f"2 | ubuntu | 3.11 | {PASS}",
    f"3 | ubuntu | 3.12 | {PASS}",
    "4 | ubuntu | 3.13 | fail | pass | test_upload_resume",
    f"5 | macos | 3.10 | {PASS}",
    f"6 | macos | 3.11 | {PASS}",
    f"7 | macos | 3.12 | {PASS}",
    f"8 | macos | 3.13 | {PASS}",
    f"9 | windows | 3.10 | {PASS}",
    "10 | windows | 3.11 | fail | fail | test_paths_roundtrip",
    "11 | windows | 3.12 | fail | fail | test_paths_roundtrip",
    "12 | windows | 3.13 | fail | fail | test_paths_roundtrip",
))
state("mx2", "table", lines(
    "Test matrix | run 4502 | branch main | started 26 Sep 08:40",
    "job | python | os | first try | re-run | failing test",
    f"1 | 3.10 | ubuntu | {PASS}",
    "2 | 3.10 | macos | fail | pass | test_import_plugins",
    f"3 | 3.10 | windows | {PASS}",
    f"4 | 3.11 | ubuntu | {PASS}",
    f"5 | 3.11 | macos | {PASS}",
    f"6 | 3.11 | windows | {PASS}",
    f"7 | 3.12 | ubuntu | {PASS}",
    f"8 | 3.12 | macos | {PASS}",
    f"9 | 3.12 | windows | {PASS}",
    "10 | 3.13 | ubuntu | fail | fail | test_import_plugins",
    "11 | 3.13 | macos | fail | fail | test_import_plugins",
    "12 | 3.13 | windows | fail | fail | test_import_plugins",
))
state("mx3", "table", lines(
    "Test matrix | run 4533 | branch fix/case-names | started 27 Sep 17:05",
    MX_HEAD,
    f"1 | windows | 3.10 | {PASS}",
    f"2 | windows | 3.11 | {PASS}",
    "3 | windows | 3.12 | fail | pass | test_cache_expiry",
    f"4 | windows | 3.13 | {PASS}",
    "5 | macos | 3.10 | fail | fail | test_case_sensitive_names",
    f"6 | macos | 3.11 | {PASS}",
    "7 | macos | 3.12 | fail | fail | test_case_sensitive_names",
    "8 | macos | 3.13 | fail | fail | test_case_sensitive_names",
    f"9 | ubuntu | 3.10 | {PASS}",
    f"10 | ubuntu | 3.11 | {PASS}",
    f"11 | ubuntu | 3.12 | {PASS}",
    f"12 | ubuntu | 3.13 | {PASS}",
))

MX_JOBS = [f"{os} / Python {v}" for os in ("ubuntu", "macos", "windows") for v in ("3.10", "3.11", "3.12", "3.13")]
ask_all("mx_common", ["What do the jobs that failed on both tries have in common?",
                      "Which of these is true of every job that failed again when re-run?"],
        ["They all run on Windows", "They all run on macOS", "They all run on Ubuntu", "They all use Python 3.13",
         "They all use Python 3.10"],
        {"mx1": "They all run on Windows", "mx2": "They all use Python 3.13", "mx3": "They all run on macOS"})
ask_all("mx_flaky", ["Which job failed on the first try but passed when re-run?",
                     "Which job only passed on its second attempt?"],
        MX_JOBS, {"mx1": "ubuntu / Python 3.13", "mx2": "macos / Python 3.10", "mx3": "windows / Python 3.12"})
ask_all("mx_ubuntu", ["Counting re-runs, did every Ubuntu job pass in the end?",
                      "Did all the ubuntu jobs end up passing once re-runs are counted?"],
        YES_NO, {"mx1": "yes", "mx2": "no", "mx3": "yes"})
ask_all("mx_reason", ["What most likely makes the jobs that failed on both tries fail?",
                      "What is the most likely cause of the failures that a re-run did not fix?"],
        ["Windows writes file paths with backslashes", "Python 3.13 no longer has something the code imports",
         "macOS treats file names that differ only in case as the same file",
         "A test depends on the network and fails at random", "The runners ran out of memory"],
        {"mx1": None, "mx2": None, "mx3": None})

# ---------------------------------------------------------------- hs: 带 CI 结果的 git 历史
# 40 个提交, 新的在前. 设计好的行手写 (special), 其余从标题池抽, CI 都 pass (除非位置在 fails 里).
AUTHORS = ["Aiko", "Hana", "Mateo", "Noor", "Priya", "Tomas"]
FILLER_SUBJECTS = [
    "Fix typo in README", "Add tests for the pager", "Update the contributor guide", "Split the settings module",
    "Rename a helper for clarity", "Bump the lint tool to 0.6", "Remove dead code in the parser",
    "Document the export flags", "Tighten types in the router", "Add a changelog entry",
    "Move fixtures into their own folder", "Log the request id on errors", "Handle empty search queries",
    "Sort imports", "Drop Python 3.8 from the matrix", "Add a health check endpoint", "Speed up the test setup",
    "Use a shared HTTP session", "Clean up unused styles", "Make the retry count configurable",
    "Add a CLI flag for dry runs", "Fix spacing on the login page", "Guard against missing headers",
    "Pin the formatter version", "Extract the pagination logic", "Add metrics for queue depth",
    "Reword the error for bad tokens", "Update screenshots in the docs", "Allow uppercase tags",
    "Hide the debug panel in production", "Add an index on created_at", "Simplify the date formatting",
    "Rename the worker pool", "Show a spinner while saving", "Add a script to seed demo data",
    "Fix wrong status code on delete", "Trim whitespace from user names", "Upgrade the test runner",
    "Add dark mode colors", "Stop logging full request bodies", "Check permissions before export",
    "Raise the default page size", "Add a migration for the tags table", "Fix a broken link in the footer",
    "Batch database inserts in the importer", "Add type hints to the models", "Improve the 404 page",
    "Handle slow responses in the webhook sender", "Split long functions in the job runner",
    "Remove the old feature flag", "Add a README for the scripts folder", "Fix a race in the job queue",
    "Use constants for status names", "Return 400 on invalid JSON", "Add an admin page for tags",
    "Improve the empty state text", "Fix a memory leak in the file watcher", "Normalise line endings",
    "Update the license year", "Make the build reproducible", "Handle unicode in file names",
]


def history(seed: str, start: tuple[int, int, int], special: dict[int, tuple[str, str]], fails: set[int],
            n: int = 40) -> tuple[list[str], list[str]]:
    """返回 (日志行, 选项 "hash 标题"), 都是新的在前. start = (日, 时, 分), 月份都是 Sep, 往前每步 35-420 分钟."""
    rng = random.Random(seed)
    pool = [s for s in FILLER_SUBJECTS if s not in {v[0] for v in special.values()}]
    rng.shuffle(pool)
    day, hour, minute = start
    t = (day * 24 + hour) * 60 + minute
    rows, opts, seen = [], [], set()
    for i in range(n):
        subject, author = special[i] if i in special else (pool.pop(), rng.choice(AUTHORS))
        while True:
            h = f"{rng.getrandbits(28):07x}"
            if h not in seen and sum(c.isalpha() for c in h) >= 2:
                break
        seen.add(h)
        d, rest = divmod(t, 24 * 60)
        rows.append(f"{h} | {d} Sep {rest // 60:02d}:{rest % 60:02d} | {author} | {subject} | "
                    f"CI {'fail' if i in fails else 'pass'}")
        opts.append(f"{h} {subject}")
        t -= rng.randint(35, 420)
    return rows, opts


HS1_ROWS, HS1 = history("hs1", (26, 17, 40), {
    3: ("Parse dates with the new library", "Hana"),
    4: ("Fix the flaky upload test", "Priya"),
    9: ('Revert "Cache rendered templates"', "Mateo"),
    16: ("Fix the broken import in the mailer", "Tomas"),
    17: ("Move mail sending into a worker", "Noor"),
    22: ("Cache rendered templates", "Aiko"),
    30: ("Cache compiled regular expressions", "Tomas"),
}, fails={0, 1, 2, 3, 17})
HS2_ROWS, HS2 = history("hs2", (19, 11, 5), {
    3: ('Revert "Batch writes to the audit table"', "Noor"),
    8: ("Upgrade the ORM to version 5", "Tomas"),
    9: ("Add a test for the CSV header", "Mateo"),
    14: ("Batch reads from the audit table", "Hana"),
    24: ("Fix the queue settings rename", "Hana"),
    25: ("Rename the queue settings", "Priya"),
    30: ("Batch writes to the audit table", "Priya"),
}, fails={0, 1, 2, 3, 4, 5, 6, 7, 8, 25})
HS3_ROWS, HS3 = history("hs3", (30, 9, 20), {
    1: ("Stream large exports", "Aiko"),
    2: ("Update the install guide", "Hana"),
    5: ("Fix the path join on Windows", "Tomas"),
    6: ("Join paths with the os helper", "Noor"),
    20: ('Revert "Inline the config loader"', "Priya"),
    27: ("Inline the theme loader", "Mateo"),
    35: ("Inline the config loader", "Aiko"),
}, fails={0, 1, 6, 7})
state("hs1", "log", lines("git log main, newest first, with the CI result of each commit", *HS1_ROWS))
state("hs2", "log", lines("git log develop, newest first, with the CI result of each commit", *HS2_ROWS))
state("hs3", "log", lines("git log trunk, newest first, with the CI result of each commit", *HS3_ROWS))

ask_each("red_start", "Which commit started the current run of failing CI results?", {
    "hs1": (HS1, HS1[3]), "hs2": (HS2, HS2[8]), "hs3": (HS3, HS3[1]),
}, judged=False)
ask_each("reverted", "Which commit was undone by a later commit in this history?", {
    "hs1": (HS1[::-1], HS1[22]), "hs2": (HS2[::-1], HS2[30]), "hs3": (HS3[::-1], HS3[35]),
}, judged=False)
ask_all("hs_author", ["Who wrote the most recent commit that passed CI?",
                      "Whose commit is the newest one with a passing CI result?"],
        AUTHORS, {"hs1": "Priya", "hs2": "Mateo", "hs3": "Hana"})

# ---------------------------------------------------------------- rv: 一次 code review 的评论串
# reviewer 提 4 点, author 逐条回, 对其中一点提出异议. 异议落在第几点、reviewer 接不接受、结局各不相同.
state("rv1", "dialogue", lines(
    "Pull request 412: Add CSV export to the reports page",
    "Hana (reviewer): src/export/csv.ts line 18: this loads every row into memory before writing. "
    "Could we stream it instead?",
    "Hana (reviewer): src/export/csv.ts line 40: the header row is missing the region column.",
    "Hana (reviewer): Please add a test for exporting an empty report.",
    "Hana (reviewer): Small one: buildCsv would read better as toCsv.",
    "Tomas (author): Streaming is done in 3f9a2ce, it now writes 500 rows at a time.",
    "Tomas (author): Region column added to the header in the same commit.",
    "Tomas (author): The empty-report test is in 81bd0ea.",
    "Tomas (author): I'd rather do the rename in its own pull request, buildCsv is called from six other files.",
    "Hana (reviewer): Fair enough, a separate pull request for the rename is fine. Approving.",
))
state("rv2", "dialogue", lines(
    "Pull request 887: Cache user permissions for five minutes",
    "Noor (reviewer): permissions.py: the cache key only uses the user id, so two tenants with the same user id "
    "would share an entry.",
    "Noor (reviewer): There is no test showing that a revoked permission stops working once the cache expires. "
    "Please add one.",
    "Noor (reviewer): Nit: CACHE_SECONDS reads better than CACHE_TTL_S.",
    "Noor (reviewer): The log line on a cache miss will be very noisy, can it go to debug level?",
    "Aiko (author): Good catch on the key, it now includes the tenant id. Pushed 5be21d0.",
    "Aiko (author): I don't think the expiry test is worth it, the cache library already has tests for that.",
    "Aiko (author): Renamed to CACHE_SECONDS.",
    "Aiko (author): Moved the log line to debug.",
    "Noor (reviewer): The library tests don't cover our revoke path, and that is the part that matters for "
    "security. I still want that test before this goes in. Requesting changes.",
))
state("rv3", "dialogue", lines(
    "Pull request 1530: Show upload progress in the file list",
    "Mateo (reviewer): FileList.tsx: polling every 200 ms seems a lot. Could we use the progress events from the "
    "upload instead?",
    "Mateo (reviewer): The progress bar has no label for screen readers.",
    'Mateo (reviewer): When an upload fails, the message just says "Error". Can it say what went wrong?',
    "Mateo (reviewer): There is a leftover console.log in useUpload.ts.",
    "Priya (author): The upload client we use doesn't emit progress events for chunked uploads, so polling is the "
    "only option. I set it to 500 ms.",
    "Mateo (reviewer): Ah, I didn't know that. 500 ms is fine then.",
    "Priya (author): Added an aria-label to the bar, and removed the console.log.",
    "Priya (author): The error message needs the new codes from the API, I'll do it tomorrow once that lands.",
    "Mateo (reviewer): I'm away from tomorrow for two weeks. Hana, could you take over this review and check the "
    "error message change when it comes in?",
    "Hana (reviewer): Sure, I'll pick it up.",
))

ask_each("open", "Which of the first reviewer's points is still not dealt with at the end of the thread?", {
    "rv1": (["Streaming rows instead of loading them all", "Adding the region column to the header",
             "Adding a test for an empty report", "Renaming buildCsv to toCsv"], "Renaming buildCsv to toCsv"),
    "rv2": (["Adding the tenant id to the cache key", "Adding a test for revoked permissions",
             "Renaming CACHE_TTL_S to CACHE_SECONDS", "Moving the cache miss log line to debug"],
            "Adding a test for revoked permissions"),
    "rv3": (["Using progress events instead of polling", "Saying what went wrong when an upload fails",
             "Giving the progress bar a label for screen readers", "Removing the leftover console.log"],
            "Saying what went wrong when an upload fails"),
})
ask_all("rv_state", ["How does the review stand at the end of the thread?", "Where has this review ended up?"],
        ["Approved and ready to merge", "Sent back to the author for more work",
         "Passed to someone else to finish reviewing"],
        {"rv1": "Approved and ready to merge", "rv2": "Sent back to the author for more work",
         "rv3": "Passed to someone else to finish reviewing"})
ask_all("rv_accept", ["Did the reviewer accept the author's reason for not making one of the requested changes?",
                      "Was the author's objection to one of the review points accepted?"],
        YES_NO, {"rv1": "yes", "rv2": "no", "rv3": "yes"})

if __name__ == "__main__":
    main(DOMAIN, LABEL, QUESTIONS, TEXTS)
