/* TEST-ONLY stand-in for Windows' icuuc.dll (Wine doesn't ship ICU). Qt only uses it for legacy text codecs. */
#include <stddef.h>
#define STUB(n) __declspec(dllexport) void* n(void) { return 0; }
__declspec(dllexport) void* ucnv_open(const char *name, int *err) { if (err) *err = 4; return 0; }
__declspec(dllexport) int ucnv_countAvailable(void) { return 0; }
STUB(ucnv_close) STUB(UCNV_FROM_U_CALLBACK_SUBSTITUTE) STUB(UCNV_TO_U_CALLBACK_SUBSTITUTE) STUB(ucnv_reset)
STUB(ucnv_getMaxCharSize) STUB(ucnv_getName) STUB(ucnv_getToUCallBack) STUB(ucnv_getFromUCallBack) STUB(ucnv_setToUCallBack)
STUB(ucnv_setFromUCallBack) STUB(ucnv_cbToUWriteUChars) STUB(ucnv_toUnicode) STUB(ucnv_cbFromUWriteUChars) STUB(ucnv_toUCountPending)
STUB(ucnv_fromUCountPending) STUB(ucnv_getStandardName) STUB(ucnv_getAvailableName) STUB(ucnv_fromUnicode)
