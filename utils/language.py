from pathlib import Path

from PyQt5.QtCore import QCoreApplication, QLocale, QTranslator


LANGUAGE_SYSTEM = "system"
LANGUAGE_EN = "en"
LANGUAGE_ZH_CN = "zh_CN"

LANGUAGE_OPTIONS = [
    (LANGUAGE_SYSTEM, "Auto"),
    (LANGUAGE_EN, "English"),
    (LANGUAGE_ZH_CN, "简体中文"),
]

TRANSLATION_FILES = {
    LANGUAGE_ZH_CN: "exvr_zh_CN.qm",
}

_translator = None
_language = LANGUAGE_SYSTEM
_effective_language = LANGUAGE_EN


def system_language():
    locale_name = QLocale.system().name()
    return LANGUAGE_ZH_CN if locale_name.lower().startswith("zh") else LANGUAGE_EN


def effective_language(language):
    if language == LANGUAGE_SYSTEM:
        return system_language()
    if language in (LANGUAGE_EN, LANGUAGE_ZH_CN):
        return language
    return system_language()


def translations_dir():
    return Path(__file__).resolve().parent.parent / "translations"


def install_language(app, language):
    global _translator, _language, _effective_language
    _language = language if language in {LANGUAGE_SYSTEM, LANGUAGE_EN, LANGUAGE_ZH_CN} else LANGUAGE_SYSTEM
    _effective_language = effective_language(_language)

    if _translator is not None:
        app.removeTranslator(_translator)
        _translator = None

    translation_file = TRANSLATION_FILES.get(_effective_language)
    if translation_file is None:
        return

    translator = QTranslator(app)
    if translator.load(str(translations_dir() / translation_file)):
        app.installTranslator(translator)
        _translator = translator


def current_language():
    return _language


def current_effective_language():
    return _effective_language


def tr(text):
    translated = QCoreApplication.translate("@default", text)
    return translated if translated else text


def _translation_sources():
    # pylupdate5 extracts tr() calls from this function. Runtime code uses the
    # same source strings through tr() and loads compiled .qm files.
    tr("Auto")
    tr("English")
    tr("Language")
    tr("ExVR {version} - Experience Virtual Reality")
    tr("Flip X")
    tr("Flip Y")
    tr("Enter IP camera URL")
    tr("Performance")
    tr("Max Performance")
    tr("Balanced")
    tr("Quality")
    tr("Aspect")
    tr("Model Provider")
    tr("Priority")
    tr("Idle")
    tr("Below Normal")
    tr("Normal")
    tr("Above Normal")
    tr("High")
    tr("Realtime")
    tr("SteamVR Installed")
    tr("SteamVR Not Installed")
    tr("Driver Update")
    tr("Updated installed components:")
    tr("Uninstall Drivers")
    tr("Install Drivers")
    tr("Start Tracking")
    tr("Stop Tracking")
    tr("Show Frame")
    tr("Hide Frame")
    tr("Only Ingame")
    tr("Currently this only applies to hotkeys and mouse input and not head movement")
    tr("window title / process name / VRChat, VRChat.exe, javaw.exe")
    tr("Reset Head")
    tr("Reset Eyes")
    tr("Reset LHand")
    tr("Reset RHand")
    tr("Head")
    tr("Face")
    tr("Tongue")
    tr("Hand")
    tr("Hand Down")
    tr("Finger Action")
    tr("Hand Return Time (s)")
    tr("Left Controller")
    tr("Right Controller")
    tr("Mouse")
    tr("Reset Hotkey")
    tr("Stop Hotkey")
    tr("Set Face")
    tr("Update Config")
    tr("Save Config")
    tr("Face Setting")
    tr("BlendShape")
    tr("Value")
    tr("Shifting")
    tr("Weight")
    tr("Max")
    tr("Enabled")
    tr("Error")
    tr("Invalid priority index")
    tr("SteamVR is not installed or could not be found.")
    tr("SteamVR is running, Please close SteamVR and try again.")
    tr("VRCFT is running, please close VRCFT and try again.")
    tr("Could not update installed drivers. Close SteamVR/VRCFaceTracking and reopen ExVR.")
    tr("Could not install/update drivers. Close SteamVR/VRCFaceTracking and try again.")
