import socket
import os
import csv
import re
from datetime import datetime
import time
import sys
import atexit

sys.stdout.reconfigure(encoding='utf-8')

# socket_5.py 只從 sys.argv[1] 讀取路徑
if len(sys.argv) > 1:
    SAVE_DIR = sys.argv[1]
    if not os.path.exists(SAVE_DIR):
        print(f"❌ 儲存路徑不存在：{SAVE_DIR}")
        exit(1)
    elif not os.access(SAVE_DIR, os.W_OK):
        print(f"❌ 儲存路徑不可寫入：{SAVE_DIR}")
        exit(1)
else:
    print("❌ 未指定儲存路徑")
    exit(1)

print(f"[INFO] Log 擷取已啟動，儲存位置：{SAVE_DIR}")

SAVE_INTERVAL = 5
os.makedirs(SAVE_DIR, exist_ok=True)

CSV_HEADER = ["Severity", "Date", "Time", "SyslogID",
              "SourceIP", "SourcePort", "DestinationIP", "DestinationPort",
              "Duration", "Bytes", "Protocol", "Action", "Description"]

log_buffer = []
file_index = 1
last_save_time = time.time()

def get_protocol(desc):
    desc = desc.lower()
    if "tcp" in desc:
        return "TCP"
    elif "udp" in desc:
        return "UDP"
    elif "icmp" in desc:
        return "ICMP"
    elif "http" in desc and "https" not in desc:
        return "HTTP"
    elif "https" in desc:
        return "HTTPS"
    elif "dns" in desc:
        return "DNS"
    elif "scan" in desc:
        return "SCAN"
    elif "flood" in desc or "rate" in desc:
        return "FLOOD"
    else:
        return "Other"

def get_action(desc):
    desc = desc.lower()
    if "teardown" in desc:
        return "Teardown"
    elif "built" in desc:
        return "Built"
    elif "deny" in desc:
        return "Deny"
    elif "translation" in desc or "translated" in desc:
        return "Translation"
    elif "login" in desc:
        return "Login"
    elif "drop" in desc:
        return "Drop"
    else:
        return "Other"

def parse_log_line(raw_line):
    timestamp = datetime.now()
    date_str = timestamp.strftime("%Y-%m-%d")
    time_str = timestamp.strftime("%H:%M:%S")

    severity_match = re.search(r'%ASA-(\d)-(\d{6})', raw_line)
    if not severity_match:
        return None

    severity = int(severity_match.group(1))
    syslog_id = severity_match.group(2)

    ip_port_match = re.search(r'(\d{1,3}(?:\.\d{1,3}){3})/(\d+)\D+(\d{1,3}(?:\.\d{1,3}){3})/(\d+)', raw_line)
    if ip_port_match:
        src_ip = ip_port_match.group(1)
        src_port = ip_port_match.group(2)
        dst_ip = ip_port_match.group(3)
        dst_port = ip_port_match.group(4)
    else:
        src_ip = src_port = dst_ip = dst_port = ""

    duration_match = re.search(r'duration (\d+:\d+:\d+)', raw_line)
    duration = duration_match.group(1) if duration_match else ""

    bytes_match = re.search(r'bytes (\d+)', raw_line)
    byte_count = bytes_match.group(1) if bytes_match else ""

    desc = raw_line.strip()
    protocol = get_protocol(desc)
    action = get_action(desc)

    return [severity, date_str, time_str, syslog_id,
            src_ip, src_port, dst_ip, dst_port,
            duration, byte_count, protocol, action, desc]

def write_log_buffer():
    global log_buffer, file_index
    filename = f"asa_logs_{file_index}.csv"
    path = os.path.join(SAVE_DIR, filename)
    with open(path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(log_buffer)
    print(f"💾 寫入 {len(log_buffer)} 筆 log 到：{filename}（目前大小 {os.path.getsize(path)//1024} KB）")
    log_buffer.clear()

def save_remaining_logs():
    if log_buffer:
        write_log_buffer()

atexit.register(save_remaining_logs)

# 開始接收 log
HOST = '0.0.0.0'
PORT = 10514

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))
print(f"✅ 開始接收 ASA syslog，儲存至：{SAVE_DIR}")

try:
    while True:
        data, addr = sock.recvfrom(4096)
        raw_log = data.decode(errors='ignore')
        print(raw_log.strip())

        parsed = parse_log_line(raw_log)
        if not parsed:
            continue

        log_buffer.append(parsed)

        # 定期儲存
        if time.time() - last_save_time >= SAVE_INTERVAL:
            filename = f"asa_logs_{file_index}.csv"
            path = os.path.join(SAVE_DIR, filename)
            # 檢查檔案大小
            if os.path.exists(path) and os.path.getsize(path) >= 500 * 1024 * 1024:
                file_index += 1
                print(f"🆕 切換到新檔案：asa_logs_{file_index}.csv")
            write_log_buffer()
            last_save_time = time.time()

except Exception as e:
    print(f"❌ 執行時發生錯誤：{str(e)}")

finally:
    sock.close()
    print("⛔ 已關閉 socket，程式結束")
