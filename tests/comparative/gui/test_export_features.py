"""
Test case: Export Tab features in ComparativeViewer.
Tests all Export tab widget interactions using SE (spin echo) data.
All interactions are done through GUI widgets using Playwright.
"""

import tempfile
from pathlib import Path

import pytest

# Timeout for GUI tests (in seconds)
GUI_TEST_TIMEOUT = 15
GUI_TEST_TIMEOUT_LONG = 30  # Extended timeout for HTML export
TASK_TIMEOUT_DEFAULT = 1000  # [ms]


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT_LONG)
def test_export_config_to_temp_location(viewer_page, se_viewer, navigate_to_tab):
    """
    Test exporting config to a temporary location via GUI.
    Load viewer without changing anything, update export config path via GUI,
    press export button via GUI, verify file is not empty.
    """
    print("\n" + "=" * 60)
    print("[test_export_config] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(se_viewer)
    print("[test_export_config] Viewer launched")

    navigate_to_tab(page, "Export")

    # Check widgets exist
    config_path_widget = page.locator(".pyeyes-export-config-path")
    config_button_widget = page.locator(".pyeyes-export-config-button")
    print(
        f"[test_export_config] Config path widget found: {config_path_widget.count() > 0}"
    )
    print(
        f"[test_export_config] Config button widget found: {config_button_widget.count() > 0}"
    )
    assert config_path_widget.count() > 0, "Config path widget should exist"
    assert config_button_widget.count() > 0, "Config button widget should exist"

    # Create temporary file path
    with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
        temp_config_path = tmp.name
        print(f"[test_export_config] Temporary config path: {temp_config_path}")

        # Update config path via GUI textarea
        ta = config_path_widget.locator("textarea")
        if ta.count() > 0:
            ta.fill(temp_config_path)
            ta.press("Tab")
            page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
            print(f"[test_export_config] Config path set via GUI: {temp_config_path}")
        else:
            raise ValueError("Textarea not found")

        # Click export config button via GUI
        config_button = config_button_widget.locator("button")
        if config_button.count() > 0:
            config_button.click()
            print("[test_export_config] Export config button clicked via GUI")
        else:
            config_button_widget.click()
            print("[test_export_config] Export config button clicked via GUI")
        page.wait_for_timeout(3000)

        # Verify file exists and is not empty
        config_file = Path(temp_config_path)
        if config_file.exists():
            file_size = config_file.stat().st_size
            print(f"[test_export_config] Exported config file size: {file_size} bytes")
            assert file_size > 0, "Exported config file should not be empty"
            print("[test_export_config] Config file exported successfully")
        else:
            raise ValueError("[test_export_config] Config file was not created")

    server.stop()
    print("[test_export_config] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT_LONG)
def test_export_html_to_temp_location(viewer_page, se_viewer, navigate_to_tab):
    """
    Test exporting HTML to a temporary location via GUI.
    Load viewer without changing anything, update HTML save path via GUI,
    press export HTML button via GUI, verify file is not empty.
    """
    print("\n" + "=" * 60)
    print("[test_export_html] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(se_viewer)
    print("[test_export_html] Viewer launched")

    navigate_to_tab(page, "Export")

    # Check widgets exist
    html_path_widget = page.locator(".pyeyes-export-html-path")
    html_button_widget = page.locator(".pyeyes-export-html-button")
    print(f"[test_export_html] HTML path widget found: {html_path_widget.count() > 0}")
    print(
        f"[test_export_html] HTML button widget found: {html_button_widget.count() > 0}"
    )
    assert html_path_widget.count() > 0, "HTML path widget should exist"
    assert html_button_widget.count() > 0, "HTML button widget should exist"

    # Create temporary file path
    with tempfile.NamedTemporaryFile(suffix=".html") as tmp:
        temp_html_path = tmp.name
        print(f"[test_export_html] Temporary HTML path: {temp_html_path}")

        # Update HTML path via GUI textarea
        ta = html_path_widget.locator("textarea")
        if ta.count() > 0:
            ta.fill(temp_html_path)
            ta.press("Tab")
            page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
            print(f"[test_export_html] HTML path set via GUI: {temp_html_path}")
        else:
            raise ValueError("Textarea not found")

        # Click export HTML button via GUI
        html_button = html_button_widget.locator("button")
        if html_button.count() > 0:
            html_button.click()
            print("[test_export_html] Export HTML button clicked via GUI")
        else:
            html_button_widget.click()
            print("[test_export_html] Export HTML button clicked via GUI")
        # HTML export takes longer
        page.wait_for_timeout(15000)

        # Verify file exists and is not empty
        html_file = Path(temp_html_path)
        if html_file.exists():
            file_size = html_file.stat().st_size
            print(f"[test_export_html] Exported HTML file size: {file_size} bytes")
            assert file_size > 0, "Exported HTML file should not be empty"
            print("[test_export_html] HTML file exported successfully")
        else:
            raise ValueError("[test_export_html] HTML file was not created")

    server.stop()
    print("[test_export_html] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_export_widgets_exist(viewer_page, se_viewer, navigate_to_tab):
    """
    Test that all Export tab widgets exist and are accessible.
    """
    print("\n" + "=" * 60)
    print("[test_export_widgets_exist] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(se_viewer)
    print("[test_export_widgets_exist] Viewer launched")

    navigate_to_tab(page, "Export")

    # List of expected widgets on Export tab
    export_widgets = [
        ("pyeyes-export-config-path-desc", "Config path description"),
        ("pyeyes-export-config-path", "Config path input"),
        ("pyeyes-export-config-button", "Export config button"),
        ("pyeyes-export-html-path", "HTML path input"),
        ("pyeyes-export-html-button", "Export HTML button"),
    ]

    for css_class, widget_name in export_widgets:
        widget = page.locator(f".{css_class}")
        count = widget.count()
        print(
            f"[test_export_widgets_exist] {widget_name} (.{css_class}): count={count}"
        )
        assert count > 0, f"{widget_name} widget should exist"

    print("[test_export_widgets_exist] All Export widgets verified")

    server.stop()
    print("[test_export_widgets_exist] Test complete")
