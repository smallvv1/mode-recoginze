#!/usr/bin/env python3
"""
海康摄像头自动重试连接脚本
重启摄像头后运行此脚本，它会持续尝试RTSP连接直到成功
"""
import cv2
import time
import sys

ip = "169.254.157.81"
creds = [
    ('admin', 'admin'),
    ('admin', '12345'),
    ('admin', '123456'),
    ('admin', '888888'),
]
paths = [
    '/Streaming/Channels/101',
    '/Streaming/Channels/102', 
    '/h264/ch1/main/av_stream',
]

max_retries = 300  # 5分钟内重试
retry_interval = 1  # 1秒重试一次

print("[*] 等待摄像头RTSP服务启动...")
print(f"[*] IP: {ip}")
print(f"[*] 将尝试 {max_retries} 次，每次间隔 {retry_interval} 秒")
print("[*] 请确保摄像头已接电并连接网线")
print()

for attempt in range(max_retries):
    print(f"[尝试 {attempt+1}/{max_retries}]", end=" ", flush=True)
    
    found = False
    for user, pwd in creds:
        if found:
            break
        for path in paths:
            url = f'rtsp://{user}:{pwd}@{ip}:554{path}'
            try:
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)
                ok, frame = cap.read()
                cap.release()
                
                if ok and frame is not None:
                    print(f"\n[✓] 连接成功!")
                    print(f"[+] RTSP URL: {url}")
                    print(f"[+] 分辨率: {frame.shape[1]}x{frame.shape[0]}")
                    print()
                    print(f"连接命令:")
                    print(f"python hikvision_ocr.py --rtsp-url \"{url}\" --lang ch")
                    sys.exit(0)
            except:
                pass
    
    if (attempt + 1) % 10 == 0:
        print(f" 已等待 {(attempt+1)*retry_interval} 秒")
    else:
        print(".", end="", flush=True)
    
    time.sleep(retry_interval)

print(f"\n[!] 未能连接到摄像头")
print(f"请检查:")
print(f"  1. 摄像头电源是否打开")
print(f"  2. 网线是否正确连接")
print(f"  3. 摄像头IP是否正确（{ip}）")
print(f"  4. 可能需要通过海康IP Installer工具启用RTSP服务")
