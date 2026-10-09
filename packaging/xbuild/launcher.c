/* Fission.exe - runs the bundled Python (runtime\python312.dll) in this process and starts app\boot.py.
   Being the process that owns the window means the taskbar, Task Manager and pinning all show "Fission". */
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#include <windows.h>
#include <shellapi.h>
#include <stdlib.h>
#include <wchar.h>

typedef int (*PyMainFn)(int, wchar_t **);

static void fail(const wchar_t *what, const wchar_t *path) {
    wchar_t msg[4096];
    _snwprintf(msg, 4096, L"Fission couldn't start: %ls\n\n%ls\n\nTry reinstalling Fission.", what, path);
    msg[4095] = 0;
    MessageBoxW(NULL, msg, L"Fission", MB_ICONERROR | MB_OK);
}

/* A GUI program normally has no stdin/stdout/stderr. Keep real redirections (files, pipes - handy for debugging),
   drop anything else (e.g. stale console handles) so Python simply sets sys.stdout to None. */
static void tidy_std_handle(DWORD id) {
    HANDLE h = GetStdHandle(id);
    if (h == NULL || h == INVALID_HANDLE_VALUE) return;
    DWORD t = GetFileType(h);
    if (t != FILE_TYPE_DISK && t != FILE_TYPE_PIPE) SetStdHandle(id, NULL);
}

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE hPrev, PWSTR cmdline, int show) {
    static wchar_t dir[32768], rt[32768], dll[32768], boot[32768];
    DWORD n = GetModuleFileNameW(NULL, dir, 32768);
    if (n == 0 || n >= 32768) { fail(L"can't find its own folder", L""); return 1; }
    wchar_t *sl = wcsrchr(dir, L'\\'); if (sl) *sl = 0;
    _snwprintf(rt, 32768, L"%ls\\runtime", dir);
    _snwprintf(dll, 32768, L"%ls\\runtime\\python312.dll", dir);
    _snwprintf(boot, 32768, L"%ls\\app\\boot.py", dir);
    if (GetFileAttributesW(boot) == INVALID_FILE_ATTRIBUTES) { fail(L"the app files are missing", boot); return 1; }

    /* a private, isolated Python: ignore any Python the user has installed */
    SetEnvironmentVariableW(L"PYTHONHOME", rt);
    SetEnvironmentVariableW(L"PYTHONPATH", NULL);
    SetEnvironmentVariableW(L"PYTHONSTARTUP", NULL);
    SetEnvironmentVariableW(L"PYTHONNOUSERSITE", L"1");
    SetEnvironmentVariableW(L"PYTHONDONTWRITEBYTECODE", L"1");
    SetDllDirectoryW(rt);
    tidy_std_handle(STD_INPUT_HANDLE); tidy_std_handle(STD_OUTPUT_HANDLE); tidy_std_handle(STD_ERROR_HANDLE);

    HMODULE py = LoadLibraryExW(dll, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!py) { fail(L"the Python runtime didn't load", dll); return 1; }
    PyMainFn py_main = (PyMainFn)GetProcAddress(py, "Py_Main");
    if (!py_main) { fail(L"the Python runtime is damaged", dll); return 1; }

    int argc = 0; wchar_t **argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    wchar_t **nargv = (wchar_t **)calloc((size_t)argc + 4, sizeof(wchar_t *));
    int k = 0;
    nargv[k++] = argv[0];
    nargv[k++] = L"-s";
    nargv[k++] = boot;
    for (int i = 1; i < argc; i++) nargv[k++] = argv[i];
    nargv[k] = NULL;
    return py_main(k, nargv);
}
