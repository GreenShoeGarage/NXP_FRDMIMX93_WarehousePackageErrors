#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <linux/fb.h>

#include "lvgl.h"
#include "ui.h"
#include "fsl_gpio.h"
#include "tensorflow/lite/c/c_api.h"

// === CONFIG ===
#define FRAME_WIDTH  320
#define FRAME_HEIGHT 240
#define SCREEN_HOR_RES 800
#define SCREEN_VER_RES 480
#define BYTES_PER_PIXEL 2 // RGB565
#define CAMERA_DEV "/dev/video0"

#define LED_GPIO GPIO5
#define LED_PIN 3U

#define MAX_LABELS 100
#define MAX_LABEL_LEN 64

// === Globals ===
static int fb_fd;
static struct fb_var_screeninfo vinfo;
static struct fb_fix_screeninfo finfo;
static uint8_t * fb_ptr = NULL;

lv_obj_t * video_img;
lv_img_dsc_t frame_img_dsc;
uint8_t * frame_buf;

int cam_fd;
struct v4l2_buffer buf;
struct v4l2_requestbuffers req;
void * buffer_start;

TfLiteModel * model;
TfLiteInterpreterOptions * options;
TfLiteInterpreter * interpreter;

char labels[MAX_LABELS][MAX_LABEL_LEN];
int label_count = 0;

// === Functions ===

void init_led() {
    gpio_pin_config_t led_config = { kGPIO_DigitalOutput, 0 };
    GPIO_PinInit(LED_GPIO, LED_PIN, &led_config);
}
void led_on() { GPIO_WritePinOutput(LED_GPIO, LED_PIN, 1); }
void led_off() { GPIO_WritePinOutput(LED_GPIO, LED_PIN, 0); }

void load_labels(const char * filename) {
    FILE * fp = fopen(filename, "r");
    if (!fp) {
        perror("Could not open labels.txt");
        exit(1);
    }
    while (fgets(labels[label_count], MAX_LABEL_LEN, fp)) {
        size_t len = strlen(labels[label_count]);
        if (len > 0 && labels[label_count][len - 1] == '\n') {
            labels[label_count][len - 1] = '\0';
        }
        label_count++;
    }
    fclose(fp);
}

void init_camera() {
    cam_fd = open(CAMERA_DEV, O_RDWR);
    if (cam_fd < 0) {
        perror("Failed to open camera");
        exit(1);
    }

    struct v4l2_format fmt;
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = FRAME_WIDTH;
    fmt.fmt.pix.height = FRAME_HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    ioctl(cam_fd, VIDIOC_S_FMT, &fmt);

    req.count = 1;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    ioctl(cam_fd, VIDIOC_REQBUFS, &req);

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    ioctl(cam_fd, VIDIOC_QUERYBUF, &buf);

    buffer_start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, cam_fd, buf.m.offset);
    ioctl(cam_fd, VIDIOC_STREAMON, &buf.type);
}

void init_tflite() {
    model = TfLiteModelCreateFromFile("../models/box_classifier.tflite");
    options = TfLiteInterpreterOptionsCreate();
    interpreter = TfLiteInterpreterCreate(model, options);
    TfLiteInterpreterAllocateTensors(interpreter);
}

void init_lvgl_video_img() {
    video_img = lv_img_create(lv_scr_act());
    lv_obj_align(video_img, LV_ALIGN_CENTER, 0, 0);

    frame_buf = malloc(FRAME_WIDTH * FRAME_HEIGHT * 3);
    frame_img_dsc.header.always_zero = 0;
    frame_img_dsc.header.w = FRAME_WIDTH;
    frame_img_dsc.header.h = FRAME_HEIGHT;
    frame_img_dsc.data_size = FRAME_WIDTH * FRAME_HEIGHT * 3;
    frame_img_dsc.header.cf = LV_IMG_CF_TRUE_COLOR;
    frame_img_dsc.data = frame_buf;

    lv_img_set_src(video_img, &frame_img_dsc);
}

void fbdev_init(void) {
    fb_fd = open("/dev/fb0", O_RDWR);
    if (fb_fd < 0) { perror("Cannot open framebuffer"); exit(1); }
    if (ioctl(fb_fd, FBIOGET_FSCREENINFO, &finfo)) { perror("Fixed info error"); exit(1); }
    if (ioctl(fb_fd, FBIOGET_VSCREENINFO, &vinfo)) { perror("Var info error"); exit(1); }

    fb_ptr = (uint8_t *)mmap(0, finfo.smem_len, PROT_READ | PROT_WRITE, MAP_SHARED, fb_fd, 0);
    if (fb_ptr == MAP_FAILED) { perror("mmap failed"); exit(1); }
}

void fbdev_flush(lv_display_t * disp_drv, const lv_area_t * area, uint8_t * color_p) {
    int32_t x, y;
    long location = 0;
    for (y = area->y1; y <= area->y2; y++) {
        for (x = area->x1; x <= area->x2; x++) {
            location = (x + vinfo.xoffset) * BYTES_PER_PIXEL +
                       (y + vinfo.yoffset) * finfo.line_length;
            memcpy(fb_ptr + location, color_p, BYTES_PER_PIXEL);
            color_p += BYTES_PER_PIXEL;
        }
    }
    lv_disp_flush_ready(disp_drv);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <label_to_match>\n", argv[0]);
        return 1;
    }
    const char * match_label = argv[1];

    BOARD_InitBootPins();
    BOARD_InitBootClocks();
    BOARD_InitDebugConsole();
    init_led();

    load_labels("../models/labels.txt");

    lv_init();
    fbdev_init();
    static lv_disp_draw_buf_t draw_buf;
    static lv_color_t buf1[SCREEN_HOR_RES * 40];
    lv_disp_draw_buf_init(&draw_buf, buf1, NULL, SCREEN_HOR_RES * 40);

    lv_disp_drv_t disp_drv;
    lv_disp_drv_init(&disp_drv);
    disp_drv.flush_cb = fbdev_flush;
    disp_drv.draw_buf = &draw_buf;
    disp_drv.hor_res = SCREEN_HOR_RES;
    disp_drv.ver_res = SCREEN_VER_RES;
    lv_disp_drv_register(&disp_drv);

    ui_init();
    init_lvgl_video_img();

    init_camera();
    init_tflite();

    while (1) {
        ioctl(cam_fd, VIDIOC_QBUF, &buf);
        ioctl(cam_fd, VIDIOC_DQBUF, &buf);
        memcpy(frame_buf, buffer_start, FRAME_WIDTH * FRAME_HEIGHT * 3);

        TfLiteTensor * input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
        memcpy(input_tensor->data.raw, frame_buf, FRAME_WIDTH * FRAME_HEIGHT * 3);
        TfLiteInterpreterInvoke(interpreter);

        TfLiteTensor * output = TfLiteInterpreterGetOutputTensor(interpreter, 0);
        float * scores = output->data.f;

        int top_class = 0;
        float top_score = scores[0];
        for (int i = 1; i < label_count; i++) {
            if (scores[i] > top_score) {
                top_score = scores[i];
                top_class = i;
            }
        }
        const char * result_label = labels[top_class];
        printf("Detected: %s (%.2f)\n", result_label, top_score);

        char status_msg[128];
        if (strcmp(result_label, match_label) == 0) {
            led_on();
            snprintf(status_msg, sizeof(status_msg), "%s package detected.", result_label);
        } else {
            led_off();
            snprintf(status_msg, sizeof(status_msg), "%s package NOT detected!", result_label);
        }
        lv_label_set_text(ui_LabelStatus, status_msg);

        lv_img_set_src(video_img, &frame_img_dsc);
        lv_timer_handler();
        usleep(33000);
    }

    close(cam_fd);
    TfLiteInterpreterDelete(interpreter);
    TfLiteModelDelete(model);
    return 0;
}
