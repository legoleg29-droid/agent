from orchestrator.core.error_classification import (
    ErrorCategory,
    RecoveryAction,
    classify_error,
    default_recovery_action,
)


def test_transient_error_text_classified_and_maps_to_retry():
    category = classify_error(error_text="Request timed out after 30s")
    assert category == ErrorCategory.TRANSIENT
    assert default_recovery_action(category) == RecoveryAction.RETRY


def test_permission_error_maps_to_replan():
    category = classify_error(error_text="Permission denied: agent lacks filesystem.write")
    assert category == ErrorCategory.PERMISSION_ERROR
    assert default_recovery_action(category) == RecoveryAction.REPLAN


def test_tool_error_maps_to_retry_or_repair():
    category = classify_error(error_text="Tool 'run_python_tests' is not registered in the ToolRegistry")
    assert category == ErrorCategory.TOOL_ERROR
    assert default_recovery_action(category) in (RecoveryAction.RETRY, RecoveryAction.REPLAN)


def test_dependency_error_maps_to_replan():
    category = classify_error(error_text="Task 't2' depends on unknown task 't1'")
    assert category == ErrorCategory.DEPENDENCY_ERROR
    assert default_recovery_action(category) == RecoveryAction.REPLAN


def test_unknown_error_text_falls_back_to_unknown_category():
    category = classify_error(error_text="something bizarre happened")
    assert category == ErrorCategory.UNKNOWN


def test_no_error_information_is_unknown():
    category = classify_error()
    assert category == ErrorCategory.UNKNOWN
