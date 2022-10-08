# ############################# Kori test builtins ############################# #
import re
from typing import Any

from kori.kori import KoriTestAction, KoriTestSubAction, KoriTestState, KoriTestError, KoriTestWarning, KoriEarlyExit, \
    KoriTestCtx


def when(action: KoriTestAction, sub_action: KoriTestSubAction) -> KoriTestAction:
    return action.also(sub_action)


def prints(*message: str, sep: str = " ", end: str = "\n"):
    expected_message = sep.join(message) + end

    def inner_prints(_ctx: KoriTestCtx, *args, **kwargs) -> tuple[KoriTestState, None]:
        actual_sep = kwargs.get("sep", " ")
        actual_end = kwargs.get("end", "\n")
        actual = actual_sep.join(args) + actual_end
        if expected_message != actual:
            return KoriTestState.fail(KoriTestError("prints", expected_message, actual)), None
        return KoriTestState.success(), None

    return KoriTestAction("prints", inner_prints, print, action_args=[expected_message])


def print_matches(regex: re.Pattern, is_err: bool = True):
    def inner_print_matches(_ctx: KoriTestCtx, *args, **kwargs) -> tuple[KoriTestState, None]:
        actual_sep = kwargs.get("sep", " ")
        actual_end = kwargs.get("end", "\n")
        actual = actual_sep.join(args) + actual_end
        if regex.match(actual) is None:
            if is_err:
                return KoriTestState.fail(KoriTestError("print_matches", regex.pattern, actual)), None
            else:
                return KoriTestState.warn(
                    KoriTestWarning("print_matches", regex.pattern, actual)
                ), None
        return KoriTestState.success(), None

    return KoriTestAction("print_matches", inner_print_matches, print, action_args=[regex])


def asks_for_input(expected_prompt: str, injected_value: str):
    def inner_asks_for_input(_ctx: KoriTestCtx, prompt: str = "") -> tuple[KoriTestState, str]:
        if expected_prompt != prompt:
            return KoriTestState.fail(
                KoriTestError("asks_for_input", expected_prompt, prompt)
            ), injected_value

        return KoriTestState.success(), injected_value

    return KoriTestAction("asks_for_input", inner_asks_for_input, input, action_args=[expected_prompt, injected_value])


def simulate_input(injected_value: str):
    def inner_simulate_input(_ctx: KoriTestCtx, _prompt: str = "") -> tuple[KoriTestState, str]:
        return KoriTestState.success(), injected_value

    return KoriTestAction("simulates_input", inner_simulate_input, input, action_args=[injected_value])


def ignore_rest():
    def ignore(_ctx: KoriTestCtx, *_args, **_kwargs) -> tuple[KoriTestState, KoriEarlyExit]:
        return KoriTestState.success(), KoriEarlyExit()

    return KoriTestAction("ignore_rest", ignore, None)


def ends():
    def inner_ends(_ctx: KoriTestCtx, *_args, fn_name: str, **_kwargs) -> tuple[KoriTestState, KoriEarlyExit]:
        return KoriTestState.fail(
            KoriTestError("ends", "No more tests", f"Called action related function: {fn_name!r}")
        ), KoriEarlyExit()

    return KoriTestAction("ends", inner_ends, None)


# ############################# Kori sub actions ############################# #


def assert_var_equals(var_name: str, expected_value: Any):
    def inner(ctx: KoriTestCtx) -> KoriTestState:
        try:
            actual_value = eval(var_name, ctx.globals_, ctx.locals_)
        except NameError:
            return KoriTestState.fail(
                KoriTestError("Var not defined",
                              f"Value {expected_value!r} for var {var_name!r}",
                              f"Var {var_name!r} was not defined")
            )
        else:
            if actual_value != expected_value:
                return KoriTestState.fail(
                    KoriTestError("Invalid value for variable",
                                  f"Value {expected_value!r} for var {var_name!r}",
                                  actual_value)
                )
            return KoriTestState.success()

    return KoriTestSubAction(action_name="assert_var_equals", action=inner, action_args=[var_name, expected_value])


def fail_and_stop():
    def inner(_ctx: KoriTestCtx):
        return KoriTestState.fail(KoriTestError(
            name="Test should fail",
            expected="Not to not fail",
            actual="Test was failed and stopped"
        ))

    return KoriTestSubAction(action_name="abort_test", action=inner)
