# ############################# Kori test builtins ############################# #
import re
from typing import Any, Callable, Optional, overload, TypeAlias

from kori.kori import KTAction, KTSubAction, KTState, KTError, KTCtx, \
    KTActionResult, KTFail

StrTest = str | re.Pattern | list[str | re.Pattern]


def remove_ignored(s: str | re.Pattern, ignored: str | None, ignore_case: bool = True):
    if ignored is None:
        return s

    s_is_regex = isinstance(s, re.Pattern)
    if isinstance(ignored, re.Pattern):
        if s_is_regex:
            s = re.compile("".join(ignored.split(s.pattern)), s.flags | (ignore_case and re.I))
        else:
            s = "".join(ignored.split(s))
    else:
        if s_is_regex:
            s = re.compile(s.pattern.replace(ignored, ""), s.flags | (ignore_case and re.I))
        else:
            s = s.replace(s, "")
    return s


def str_match(expected: StrTest, actual: str, ignore: str | re.Pattern | None = re.compile("\n| *"),
              ignore_case: bool = True) -> tuple[bool, tuple[str, str]]:
    original_expected = expected
    original_actual = actual
    actual = remove_ignored(actual, ignore)

    if isinstance(expected, list):
        expected = [remove_ignored(s, ignore, ignore_case) for s in expected]

        def compare_s(s, actual_s): return s.lower() in actual_s.lower() if ignore_case else s in actual_s

        matches = all(compare_s(s, actual) if isinstance(s, str) else s.search(actual) is not None for s in expected)

        return matches, (original_expected, original_actual)

    expected = remove_ignored(expected, ignore)

    expected_is_regex = isinstance(expected, re.Pattern)
    if expected_is_regex:
        return expected.search(actual) is not None, (original_expected.pattern, original_actual)

    # expected_message is not a regex
    return expected == actual, (original_expected, original_actual)


def one_of(*sub_actions: KTSubAction) -> KTSubAction:
    def inner_one_of(ctx: KTCtx):
        for sub_action in sub_actions:
            result = sub_action.action(ctx)
            if result.is_success():
                return result
        return KTState.fail(KTFail("one_of", "One of the action to match", "None have matched"))

    return KTSubAction("one_of", inner_one_of, action_args=[sub_action.action_name for sub_action in sub_actions])


def every(*sub_actions: KTSubAction) -> KTSubAction:
    def inner_one_of(ctx: KTCtx):
        for sub_action in sub_actions:
            result = sub_action.action(ctx)
            if result.is_failure():
                break
        else:
            return KTState.success()
        return KTState.fail(KTFail("every", "All actions must match", "At least one did not match"))

    return KTSubAction("one_of", inner_one_of, action_args=[sub_action.action_name for sub_action in sub_actions])


def when(action: KTAction, sub_action: KTSubAction) -> KTAction:
    return action.also(sub_action)


def prints(expected_message: StrTest, ignore: StrTest = re.compile(r"\n| *")) -> KTAction:
    def inner_prints(_ctx: KTCtx, *args, **kwargs) -> KTActionResult:
        actual_sep = kwargs.get("sep", " ")
        actual_end = kwargs.get("end", "\n")
        actual = actual_sep.join(map(str, args)) + actual_end

        matches, expected_actual = str_match(expected_message, actual, ignore)

        if not matches:
            return KTActionResult.fail(KTFail("printed message did not match", *expected_actual), None)
        return KTActionResult.success(None)

    return KTAction("prints", inner_prints, [print], action_args=[expected_message, ignore])


def asks_for_input(expected_prompt: StrTest, injected_value: str) -> KTAction:
    def inner_asks_for_input(_ctx: KTCtx, prompt: str = "") -> KTActionResult:
        matches, expected_actual = str_match(expected_prompt, prompt)
        if not matches:
            return KTActionResult.fail(KTFail("asks_for_input", *expected_actual), injected_value)

        return KTActionResult.success(injected_value)

    return KTAction("asks_for_input", inner_asks_for_input, [input],
                    action_args=[expected_prompt, injected_value])


def simulate_randint(expected_range: range, result: int) -> KTAction:
    from random import randint, randrange, choice

    def inner_simulate_randint(ctx: KTCtx, *args, **_kwargs) -> KTActionResult:
        if ctx.current_fn.__name__ == "choice":
            choices = args[0]
            start, stop = min(choices), max(choices)
            if not all(expected == actual for expected, actual in zip(range(start, stop + 1), choices)):
                return KTActionResult.fail(KTFail(
                    "Invalid range",
                    f"start={expected_range.start}, stop={expected_range.stop}",
                    f"{start=}, {stop=}, with {choices=}"
                ), result)
        else:
            start, stop = args

        if ctx.current_fn.__name__ == "randrange":
            stop -= 1

        if expected_range.start != start or expected_range.stop != stop:
            return KTActionResult.fail(KTFail(
                "Invalid range",
                f"start={expected_range.start}, stop={expected_range.stop}",
                f"{start=}, {stop=}"
            ), result)

        return KTActionResult.success(result)

    return KTAction("simulate_randint", inner_simulate_randint,
                    [choice, randint, randrange], action_args=[result])


@overload
def capture_stdout_until_input(returned: str) -> KTAction:
    ...


@overload
def capture_stdout_until_input(returned: Callable[[KTCtx], str]) -> KTAction:
    ...


@overload
def capture_stdout_until_input(returned: None) -> KTAction:
    ...


def capture_stdout_until_input(returned):
    def inner_capture_stdout_until(ctx: KTCtx, *args, **kwargs) -> KTActionResult:
        fn_name = ctx.current_fn.__name__

        if fn_name == "print":
            sep = kwargs.get("sep", " ")
            end = kwargs.get("end", "\n")
            ctx.write_stdout(sep.join(map(str, args)) + end)
            return KTActionResult.success(None, is_done=False)

        elif fn_name == "input":
            if returned is None:
                return KTActionResult.success(None, redo_with_next_action=True)
            result: str = returned if isinstance(returned, str) else returned(ctx)
            ctx.write_stdout(args[0] if len(args) > 0 else "")
            return KTActionResult.success(result)

        else:
            return KTActionResult.fail(KTFail("capture_stdout_until_input", "Unreachable", "Reached"), None)

    return KTAction("capture_stdout_until_input",
                    inner_capture_stdout_until,
                    [print, input],
                    action_args=[returned])


def simulate_input(injected_value: str):
    def inner_simulate_input(_ctx: KTCtx, _prompt: str = "") -> KTActionResult:
        return KTActionResult.success(injected_value)

    return KTAction("simulates_input", inner_simulate_input, [input], action_args=[injected_value])


def ignore_rest():
    def ignore(_ctx: KTCtx, *_args, **_kwargs) -> KTActionResult:
        return KTActionResult.success(None, end_test=True)

    return KTAction("ignore_rest", ignore, None)


def ends():
    def inner_ends(ctx: KTCtx, *_args, **_kwargs) -> KTActionResult:
        fn_name = ctx.current_fn.__name__
        return KTActionResult.fail(
            KTFail("ends", "No more tests", f"Called action related function: {fn_name!r}"),
            None,
            end_test=True
        )

    return KTAction("ends", inner_ends, None)


# ############################# Kori sub actions ############################# #


def assert_var_equals(var_name: str, expected_value: Any):
    def inner(ctx: KTCtx) -> KTState:
        try:
            actual_value = eval(var_name, ctx.globals_, ctx.locals_)
        except NameError as e:
            return KTState.err(
                KTError(f"Var {var_name!r} not defined",
                        expected_value, str(e))
            )
        else:
            if actual_value != expected_value:
                return KTState.err(
                    KTError(f"Invalid value for variable {var_name!r}",
                            expected_value, actual_value)
                )
            return KTState.success()

    return KTSubAction(action_name="assert_var_equals", action=inner, action_args=[var_name, expected_value])


def assert_stdout_matches(expected_stdout: StrTest, ignore: StrTest | None = re.compile(r"\n| *"),
                          ignore_case: bool = True, flush_stdout: bool = True):
    def inner_assert_stdout_matches(ctx: KTCtx) -> KTState:
        stdout = ctx.read_stdout() if flush_stdout else ctx.peek_stdout()

        matches, expected_actual = str_match(expected_stdout, stdout, ignore, ignore_case)

        if not matches:
            return KTState.err(KTError("assert_stdout_matches", *expected_actual))

        return KTState.success()

    return KTSubAction("assert_stdout_matches", inner_assert_stdout_matches,
                       [expected_stdout, ignore, f"ignore_case={ignore_case}"])


def assert_stdout_contains(*sub_str: StrTest):
    def inner_assert_stdout_matches(ctx: KTCtx) -> KTState:
        stdout = ctx.read_stdout()

        if not all((s in stdout) if isinstance(s, str) else s.match(stdout) is not None for s in sub_str):
            return KTState.err(KTError("assert_stdout_contains", f"{sub_str!r} in printed message", stdout))

        return KTState.success()

    return KTSubAction("assert_stdout_contains", inner_assert_stdout_matches, [sub_str])


def fail_and_stop():
    def inner(_ctx: KTCtx):
        return KTState.err(KTError(
            name="Test should fail",
            expected="Not to not fail",
            actual="Test was failed and stopped"
        ))

    return KTSubAction(action_name="abort_test", action=inner)
