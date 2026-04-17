import os
import sys
import datetime
import time
import argparse
import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from MvCameraControl_class import *
from CameraParams_header import *
 
 
#-------------------opencv操作部分--------------------------------------
def opencv_action(img):
    #自己定义操作
    result_img = img
    return result_img


def try_set_max_resolution(camera):
    width_param = MVCC_INTVALUE()
    height_param = MVCC_INTVALUE()

    ret_w = camera.MV_CC_GetIntValue("Width", width_param)
    ret_h = camera.MV_CC_GetIntValue("Height", height_param)

    if ret_w == 0 and hasattr(width_param, "nMax") and int(width_param.nMax) > 0:
        camera.MV_CC_SetIntValue("Width", int(width_param.nMax))
    if ret_h == 0 and hasattr(height_param, "nMax") and int(height_param.nMax) > 0:
        camera.MV_CC_SetIntValue("Height", int(height_param.nMax))


def try_set_best_pixel_format(camera):
    # 优先彩色 BayerRG8，失败则尝试 BayerGB8
    ret = camera.MV_CC_SetEnumValue("PixelFormat", 17301513)
    if ret != 0:
        camera.MV_CC_SetEnumValue("PixelFormat", 17301514)


def get_mode_config(mode_name):
    mode_table = {
        "ultrafast": {
            "exposure_time": 22000.0,
            "gain": 10.0,
            "target_frame_count": 1,
            "warmup_frames": 0,
            "jpeg_quality": 88,
            "score_scale": 0.0,
            "use_first_frame": True,
        },
        "fast": {
            "exposure_time": 10000.0,
            "gain": 7.0,
            "target_frame_count": 3,
            "warmup_frames": 0,
            "jpeg_quality": 90,
            "score_scale": 0.0,
            "use_first_frame": True,
        },
        "bright": {
            "exposure_time": 26000.0,
            "gain": 12.0,
            "target_frame_count": 12,
            "warmup_frames": 3,
            "jpeg_quality": 95,
            "score_scale": 0.25,
            "use_first_frame": False,
        },
        "quality": {
            "exposure_time": 32000.0,
            "gain": 10.0,
            "target_frame_count": 20,
            "warmup_frames": 4,
            "jpeg_quality": 98,
            "score_scale": 0.33,
            "use_first_frame": False,
        },
    }
    return mode_table[mode_name]


parser = argparse.ArgumentParser(description="HK camera capture tool")
parser.add_argument("--mode", choices=["ultrafast", "fast", "bright", "quality"], default="fast")
parser.add_argument("--count", type=int, default=1, help="拍摄张数（相机保持连接）")
args = parser.parse_args()
mode_cfg = get_mode_config(args.mode)

# 记录总程序开始时间
start_time = time.time()
 
#-----------------------海康相机设置部分---------------------------------------

if hasattr(MvCamera, "MV_CC_Initialize"):
    ret = MvCamera.MV_CC_Initialize()
    if ret != 0:
        print(f"初始化SDK失败，错误码: {ret}")
        exit()
 
# 枚举设备
deviceList = MV_CC_DEVICE_INFO_LIST()
n_layer_type = MV_GIGE_DEVICE | MV_USB_DEVICE
gntl_cameralink_device = globals().get("MV_GENTL_CAMERALINK_DEVICE")
if gntl_cameralink_device is not None:
    n_layer_type |= gntl_cameralink_device
else:
    mv_cameralink_device = globals().get("MV_CAMERALINK_DEVICE")
    if mv_cameralink_device is not None:
        n_layer_type |= mv_cameralink_device

ret = MvCamera.MV_CC_EnumDevices(n_layer_type, deviceList)
if ret != 0:
    print(f"枚举设备失败，错误码: {ret}")
    exit()
 
print(f"找到 {deviceList.nDeviceNum} 台设备")
if deviceList.nDeviceNum == 0:
    exit()
 
stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
 
camera = MvCamera()
 
ret = camera.MV_CC_CreateHandle(stDeviceList)
if ret != 0:
    print(f"创建句柄失败，错误码: {ret}")
    exit()
 
# 打开设备（使用已创建的句柄）
ret = -1
for try_index in range(5):
    ret = camera.MV_CC_OpenDevice()
    if ret == 0:
        break
    time.sleep(0.4)

if ret != 0:
    print(f"打开设备失败，错误码: {ret}")
    print("请关闭其他占用相机的软件（如 MVS 客户端/其他 Python 进程）后重试")
    camera.MV_CC_DestroyHandle()
    exit()

try_set_max_resolution(camera)
try_set_best_pixel_format(camera)
 
 
 
# 获取相机参数
width = c_uint()
height = c_uint()
payload_size = c_uint()
stParam = MVCC_INTVALUE()
 
ret = camera.MV_CC_GetIntValue("PayloadSize", stParam)
if ret != 0:
    print(f"获取PayloadSize失败，错误码: {ret}")
    exit()
payload_size.value = stParam.nCurValue
 
# 获取宽度
ret = camera.MV_CC_GetIntValue("Width", stParam)
if ret != 0:
    print(f"获取宽度失败，错误码: {ret}")
    exit()
width.value = stParam.nCurValue
 
# 获取高度
ret = camera.MV_CC_GetIntValue("Height", stParam)
if ret != 0:
    print(f"获取高度失败，错误码: {ret}")
    exit()
height.value = stParam.nCurValue
 
print(width.value,height.value)

# 关闭自动曝光（部分机型可能不支持，失败时忽略）
camera.MV_CC_SetEnumValue("ExposureAuto", 0)
camera.MV_CC_SetEnumValue("GainAuto", 0)

# 拍照参数（按模式切换）
exposure_time = mode_cfg["exposure_time"]  # 单位：微秒
ret = camera.MV_CC_SetFloatValue("ExposureTime", exposure_time)
if ret != 0:
    print(f"设置曝光失败，错误码: {ret}")

ret = camera.MV_CC_SetFloatValue("Gain", mode_cfg["gain"])
if ret != 0:
    print(f"设置增益失败，错误码: {ret}")
 
 
 
# 开始抓图
ret = camera.MV_CC_StartGrabbing()
if ret != 0:
    print(f"开始抓图失败，错误码: {ret}")
    exit()
 
# 分配缓冲区
data_buf = (c_ubyte * payload_size.value)()
 
stFrameInfo = MV_FRAME_OUT_INFO_EX()
 
#-----------------------------------------------运行部分---------------------------

print(f"当前模式: {args.mode}")
print(f"模式参数: exposure={exposure_time}us, gain={mode_cfg['gain']}, frames={mode_cfg['target_frame_count']}")
print(f"拍摄次数: {args.count}")

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

capture_results = []  # 记录每次拍摄结果

try:
    for capture_idx in range(1, args.count + 1):
        # 自动抓取若干帧，按清晰度选最优并保存
        target_frame_count = mode_cfg["target_frame_count"]
        warmup_frames = mode_cfg["warmup_frames"]
        captured_count = 0
        best_score = -1.0
        best_frame = None
        best_brightness = 0.0

        output_name = f"capture_clear_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        output_path = os.path.join(CURRENT_DIR, output_name)

        print(f"\n[{capture_idx}/{args.count}] 开始拍摄...")
 
        while captured_count < target_frame_count:
            data_buf = (c_ubyte * payload_size.value)()
            ret = camera.MV_CC_GetOneFrameTimeout(
                byref(data_buf),
                payload_size.value,
                stFrameInfo,
                1000
            )
 
            if ret == 0:
                #print(f"获取到帧: 宽度={stFrameInfo.nWidth}, 高度={stFrameInfo.nHeight}, "f"像素格式={stFrameInfo.enPixelType}, 帧大小={stFrameInfo.nFrameLen}")
 
                frame = np.frombuffer(data_buf, dtype=np.uint8, count=stFrameInfo.nFrameLen)
                actual_width = stFrameInfo.nWidth
                actual_height = stFrameInfo.nHeight
                pixel_type = stFrameInfo.enPixelType
 
                if pixel_type == 17301505:  # Mono8
                    expected_size = actual_width * actual_height
                    if len(frame) != expected_size:
                        print(f"数据大小不匹配: 期望 {expected_size}, 实际 {len(frame)}")
                        continue
 
                    frame = frame.reshape((actual_height, actual_width))
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif pixel_type == 17301513:  # BayerRG8
                    expected_size = actual_width * actual_height
                    if len(frame) != expected_size:
                        print(f"数据大小不匹配: 期望 {expected_size}, 实际 {len(frame)}")
                        continue
 
                    frame = frame.reshape((actual_height, actual_width))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)
                elif pixel_type == 17301514:  # BayerGB8
                    expected_size = actual_width * actual_height
                    if len(frame) != expected_size:
                        print(f"数据大小不匹配: 期望 {expected_size}, 实际 {len(frame)}")
                        continue
 
                    frame = frame.reshape((actual_height, actual_width))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BayerGB2BGR)
                elif len(frame) == actual_width * actual_height * 3:
                    frame = frame.reshape((actual_height, actual_width, 3))
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    print(f"不支持的像素格式: {stFrameInfo.enPixelType}")
                    break

                frame = opencv_action(frame)

                captured_count += 1
                if mode_cfg.get("use_first_frame", False):
                    best_frame = frame.copy()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    best_score = 100.0
                    best_brightness = float(np.mean(gray))
                    break
                elif captured_count > warmup_frames:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    score_scale = mode_cfg["score_scale"]
                    gray_small = cv2.resize(gray, (0, 0), fx=score_scale, fy=score_scale, interpolation=cv2.INTER_AREA)
                    sharpness = cv2.Laplacian(gray_small, cv2.CV_64F).var()
                    brightness = float(np.mean(gray))
                    target_brightness = 120.0
                    brightness_factor = max(0.0, 1.0 - abs(brightness - target_brightness) / target_brightness)
                    score = sharpness * (0.55 + 0.45 * brightness_factor)
                    if score > best_score:
                        best_score = score
                        best_frame = frame.copy()
                        best_brightness = brightness
 
                cv2.imshow("Camera", frame)
                cv2.setWindowTitle(
                    "Camera",
                    f"Camera | frame={captured_count}/{target_frame_count} | score={best_score:.2f}"
                )
 
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
 
            else:
                print(f"获取图像失败，错误码: {ret}")
                break

        if best_frame is not None:
            ok = cv2.imwrite(output_path, best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), mode_cfg["jpeg_quality"]])
            if ok:
                saved_height, saved_width = best_frame.shape[:2]
                elapsed_time = time.time() - start_time
                print(f"  [{capture_idx}] 已保存: {output_path}")
                print(f"  分辨率: {saved_width} x {saved_height} | 亮度: {best_brightness:.2f} | 耗时: {elapsed_time:.2f}秒")
                capture_results.append({"path": output_path, "brightness": best_brightness, "time": elapsed_time})
            else:
                print(f"  [{capture_idx}] 保存图像失败")
        else:
            print(f"  [{capture_idx}] 未获取到可保存的有效图像")

    print(f"\n总计拍摄: {len(capture_results)} 张")
    if capture_results:
        avg_brightness = sum(r["brightness"] for r in capture_results) / len(capture_results)
        final_time = time.time() - start_time
        print(f"平均亮度: {avg_brightness:.2f}")
        print(f"总耗时: {final_time:.2f} 秒 | 平均每张: {final_time/len(capture_results):.2f} 秒")
 
 
 
finally:
    # 停止抓图
    camera.MV_CC_StopGrabbing()
    # 关闭设备
    camera.MV_CC_CloseDevice()
    # 销毁句柄
    camera.MV_CC_DestroyHandle()
    # 反初始化SDK（如果当前版本支持）
    if hasattr(MvCamera, "MV_CC_Finalize"):
        MvCamera.MV_CC_Finalize()
    # 销毁窗口
    cv2.destroyAllWindows()