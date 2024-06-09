
// #include <WiFi.h>
// #include <WebServer.h>
// new OTA
// #include <ArduinoOTA.h>
// const char* ssid = "hello";
// const char* password = "00000000";

//lib for model
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "c_model_name2.h"

// lib esp32
#include "esp_camera.h"
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// 4 for flash led or 33 for normal led

#define LED_GPIO_NUM       4

// frame image
constexpr int kImageWidth = 32;
constexpr int kImageHeight = 32;
constexpr int kImageChannels = 1; 
uint8_t image_data[kImageWidth * kImageHeight * kImageChannels];

// ----------------------

namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  constexpr int kTensorArenaSize = 32 * 1024;
  //constexpr int kTensorArenaSize = 16 * 1024;  // giảm từ 32KB xuống 16KB

  uint8_t tensor_arena[kTensorArenaSize];
}

// tflite 

#if CONFIG_FREERTOS_UNICORE
#define ARDUINO_RUNNING_CORE 0
#else
#define ARDUINO_RUNNING_CORE 1
#endif

#define ANALOG_INPUT_PIN 2

#ifndef LED_BUILTIN
  #define LED_BUILTIN 14
#endif

// chân của siêu âm
constexpr int trigPin = 13; 
constexpr int echoPin = 12; 

constexpr int ledLock1 = 2; 
constexpr int ledLock2 = 14; 
constexpr int ledLock = 15; 
// Define two tasks
void TaskCheck( void *param );
void TaskPredict( void *param );

TaskHandle_t taskHandleCheckPeople; 
TaskHandle_t taskHandlePredict;



void setup() {
  Serial.begin(115200);

  //   Serial.println("Booting");
  // WiFi.mode(WIFI_STA);
  // WiFi.begin(ssid, password);
  // while (WiFi.waitForConnectResult() != WL_CONNECTED) {
  //   Serial.println("Connection Failed! Rebooting...");
  //   delay(5000);
  //   ESP.restart();
  // }

  // ArduinoOTA
  //   .onStart([]() {
  //     String type;
  //     if (ArduinoOTA.getCommand() == U_FLASH)
  //       type = "sketch";
  //     else // U_SPIFFS
  //       type = "filesystem";

  //     Serial.println("Start updating " + type);
  //   })
  //   .onEnd([]() {
  //     Serial.println("\nEnd");
  //   })
  //   .onProgress([](unsigned int progress, unsigned int total) {
  //     Serial.printf("Progress: %u%%\r", (progress / (total / 100)));
  //   })
  //   .onError([](ota_error_t error) {
  //     Serial.printf("Error[%u]: ", error);
  //     if (error == OTA_AUTH_ERROR) Serial.println("Auth Failed");
  //     else if (error == OTA_BEGIN_ERROR) Serial.println("Begin Failed");
  //     else if (error == OTA_CONNECT_ERROR) Serial.println("Connect Failed");
  //     else if (error == OTA_RECEIVE_ERROR) Serial.println("Receive Failed");
  //     else if (error == OTA_END_ERROR) Serial.println("End Failed");
  //   });

  // ArduinoOTA.begin();

  // Serial.println("Ready");
  // Serial.print("IP address: ");
  // Serial.println(WiFi.localIP());

  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(ledLock1, OUTPUT);
  pinMode(ledLock2, OUTPUT);
  pinMode(ledLock, OUTPUT);
  pinMode(LED_GPIO_NUM, OUTPUT);
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  // Initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
Serial.println("ok1");
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  model = tflite::GetModel(c_model_name2);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
Serial.println("ok2");
  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }
Serial.println("ok3");
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  xTaskCreatePinnedToCore(
    TaskPredict, //task se thuc thi
    "Task Predict",//ten
    102400, /kich thuoc ngan sep
    NULL,//tham so
    2,//do uu tien
    &taskHandlePredict,//tro den TaskHandle_t(con tro tro den task)
    0//loi cpu
  );

  xTaskCreatePinnedToCore(
      TaskCheck,
      "Task Check People",
      1024, 
      NULL,
      1,
      &taskHandleCheckPeople,
      1);


  Serial.println("ok4");

  vTaskSuspend(taskHandlePredict);


}




// ---------------------------------------------------

void resize_image_to_32x32(uint8_t* input, uint8_t* output, int inputWidth, int inputHeight) {
  float scaleWidth = inputWidth / (float)kImageWidth;
  float scaleHeight = inputHeight / (float)kImageHeight;
  for (int y = 0; y < kImageHeight; y++) {
    for (int x = 0; x < kImageWidth; x++) {
      int srcX = (int)(x * scaleWidth);
      int srcY = (int)(y * scaleHeight);
      srcX = min(srcX, inputWidth - 1);
      srcY = min(srcY, inputHeight - 1);
      int inputIndex = (srcY * inputWidth) + srcX;
      int outputIndex = (y * kImageWidth) + x;
      output[outputIndex] = input[inputIndex];
    }
  }
}

// ---------------------------------------------------
// void resize_image_to_32x32_grayscale(uint8_t* input, uint8_t* output, int inputWidth, int inputHeight) {
//   // Tỉ lệ thu nhỏ hình ảnh
//   float scaleWidth = inputWidth / (float)kImageWidth;
//   float scaleHeight = inputHeight / (float)kImageHeight;

//   // Vì hình ảnh gốc có 3 kênh màu (RGB), mỗi pixel có 3 giá trị.
//   const int bytesPerPixel = 3;s

//   // Duyệt qua từng pixel của hình ảnh đầu ra
//   for (int y = 0; y < kImageHeight; y++) {
//     for (int x = 0; x < kImageWidth; x++) {
//       // Tìm vị trí tương ứng trên hình ảnh gốc
//       int srcX = (int)(x * scaleWidth);
//       int srcY = (int)(y * scaleHeight);

//       // Đảm bảo không vượt quá ranh giới của pixel trong hình ảnh gốc
//       srcX = min(srcX, inputWidth - 1);
//       srcY = min(srcY, inputHeight - 1);

//       // Tính chỉ số cho mảng đầu vào dùng cho hình ảnh màu
//       int inputIndex = (srcY * inputWidth + srcX) * bytesPerPixel;

//       // Trích xuất giá trị R, G, B
//       uint8_t red = input[inputIndex];       // R
//       uint8_t green = input[inputIndex + 1]; // G
//       uint8_t blue = input[inputIndex + 2];  // B

//       // Chuyển đổi sang màu xám theo trọng số
//       uint8_t grayScale = (uint8_t)(0.299f * red + 0.587f * green + 0.114f * blue);

//       // Tính chỉ số cho mảng đầu ra dùng cho hình ảnh xám
//       int outputIndex = (y * kImageWidth) + x;

//       // Gán giá trị màu xám sang hình ảnh đầu ra
//       output[outputIndex] = grayScale;
//     }
//   }
// }

// ---------------------------------------------------

void loop(){
  //ArduinoOTA.handle();
}

// ---------------------------------------------------


void TaskCheck(void *pvParameters) {
  (void) pvParameters;
  digitalWrite(ledLock, HIGH);

  for (;;) { 
    Serial.println("TaskCheck: Running");

    long duration, distance;
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);
    duration = pulseIn(echoPin, HIGH);

    distance = (duration / 2) / 29.1; 
    Serial.print("Measured Distance: ");
    Serial.println(distance);

    if (distance <= 20) {
      Serial.println("Distance <= 20 cm: Starting TaskPredict");

      if (taskHandlePredict != NULL) { 
        vTaskResume(taskHandlePredict);
        vTaskSuspend(taskHandleCheckPeople);
      } else {
        Serial.println("TaskPredict is not created or handle is invalid.");
      }
    } else {
      Serial.println("Distance > 20 cm.");
    }

    vTaskDelay(1000 / portTICK_PERIOD_MS);
  }
}
  // pinMode(ledLock1, OUTPUT);
  // pinMode(ledLock2, OUTPUT);
  // pinMode(ledLock, OUTPUT);
  // pinMode(LED_GPIO_NUM, OUTPUT);
void TaskPredict(void *pvParameters) {
  (void)pvParameters;
  int count = 0; 
  double class1 = 0;
  double class2 = 0;
  
  Serial.println("Camera");

  for (;;) {
    // if(count == 0){
    //   digitalWrite(LED_GPIO_NUM, HIGH);
    // }
     Serial.println("Task2");
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      return;
    }

    resize_image_to_32x32(fb->buf, image_data, fb->width, fb->height);

    float image_normalized[kImageWidth * kImageHeight];
    for (int i = 0; i < kImageWidth * kImageHeight; i++) {
      image_normalized[i] = image_data[i] / 255.0f;
    }

    if (model_input->type != kTfLiteFloat32) {
      Serial.println("Input tensor type is incorrect, should be float!");
      return;
    }
    memcpy(model_input->data.f, image_normalized, sizeof(image_normalized));
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on x_val");
      return;
    }
    Serial.print("Prediction for class 1: ");
    Serial.println(model_output->data.f[0]);
    class1 += model_output->data.f[0];
    Serial.print("Prediction for class 2: ");
    Serial.println(model_output->data.f[1]);
    class2 += model_output->data.f[1];
    esp_camera_fb_return(fb);
  count ++;
    if (count >= 10) {
      digitalWrite(LED_GPIO_NUM, LOW);
      if (class1 > class2){
        digitalWrite(ledLock1, HIGH);
        delay(1000);
        digitalWrite(ledLock1, LOW);
      }else {
        digitalWrite(ledLock2, HIGH);
        delay(1000);
        digitalWrite(ledLock2, LOW);
      }
      count = 0;
      class1 = 0;
      class2 = 0;

      Serial.println("stop");
      vTaskResume(taskHandleCheckPeople);
      vTaskSuspend(taskHandlePredict);
    }
     vTaskDelay(1000 / portTICK_PERIOD_MS);
  }
}
