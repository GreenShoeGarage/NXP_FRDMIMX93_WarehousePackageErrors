#include "lvgl.h"
#include "ui.h"

lv_obj_t * ui_packageSizeDetected;

void ui_init(void) {
    lv_obj_t * scr = lv_scr_act();
    ui_packageSizeDetected = lv_label_create(scr);
    lv_label_set_text(ui_packageSizeDetected "Waiting...");
    lv_obj_align(ui_packageSizeDetected, LV_ALIGN_TOP_MID, 0, 10);
}
