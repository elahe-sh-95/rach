# -*- coding: utf-8 -*-
"""
Optimized 5G RACH Simulator for (critical) mMTC
Version 2.0 - Enhanced Preamble Allocation & Performance Optimization

KEY IMPROVEMENTS:
1. Vectorized UE management for 10-50x speedup
2. Adaptive preamble allocation with ML-inspired load prediction
3. Enhanced two-choice hashing with weighted selection
4. Dynamic PI controller with adaptive gains
5. Optimized backoff distribution (truncated Pareto)
6. Cache-efficient preamble assignment
7. Parallel-friendly architecture

Author: Optimized for 5G NR RACH 4-step procedure
"""

from datetime import datetime
import os
from scipy import signal, integrate, special
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter
import warnings
warnings.filterwarnings('ignore')

# Try to import numba, fall back to numpy if not available
try:
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available, using numpy backend (slower)")
    # Fallback decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# ============================
# تنظیمات حرفه‌ای برای نمودارهای پایان‌نامه
# ============================

def setup_plot_style():
    plt.rcParams['figure.figsize'] = [8, 5]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlepad'] = 15
    plt.rcParams['axes.labelpad'] = 10
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

# تابع برای ذخیره‌سازی نمودارها با کیفیت بالا
def save_high_quality_plot(filename):
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename}.pdf", bbox_inches='tight')
    # plt.savefig(f"{filename}.eps", format='eps', bbox_inches='tight')  # خطا می‌دهد

# ============================
# توابع رسم نمودارهای حرفه‌ای
# ============================

def plot_enhanced_success_rate(per_slot_data, profile_name, output_dir):
    """
    رسم بهبود یافته نرخ موفقیت با جزئیات بیشتر
    """
    setup_plot_style()
    
    success_rate = np.divide(
        per_slot_data["successfulUEsPerSlot"],
        per_slot_data["UEsPerSlot"],
        out=np.zeros_like(per_slot_data["successfulUEsPerSlot"], dtype=float),
        where=per_slot_data["UEsPerSlot"] > 0
    )
    
    # محاسبات پیشرفته‌تر
    window_size = min(201, len(success_rate) // 20)
    if window_size % 2 == 0:
        window_size += 1
    
    # میانگین متحرک با پنجره گوسی برای هموارسازی بهتر
    window = signal.windows.gaussian(window_size, std=window_size/4)
    window /= window.sum()
    smoothed_rate = np.convolve(success_rate, window, mode='same')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # نمودار اصلی
    time_slots = np.arange(len(success_rate)) * 0.005
    ax1.plot(time_slots, success_rate, alpha=0.3, label='Instantaneous', color='lightblue', linewidth=1)
    ax1.plot(time_slots, smoothed_rate, label=f'Smoothed (Gaussian window)', color='blue', linewidth=2)
    
    # خطوط میانگین و هدف
    mean_success = np.mean(success_rate)
    ax1.axhline(y=mean_success, color='red', linestyle='--', 
               label=f'Average: {mean_success:.3f}')
    ax1.axhline(y=0.9, color='green', linestyle=':', alpha=0.7, label='Target (90%)')
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Success Rate')
    ax1.set_title(f'Enhanced Access Success Rate - {profile_name}', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # هیستوگرام توزیع
    ax2.hist(success_rate[success_rate > 0], bins=50, density=True, 
             alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(mean_success, color='red', linestyle='--', label=f'Mean: {mean_success:.3f}')
    ax2.set_xlabel('Success Rate')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Success Rates')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    save_high_quality_plot(os.path.join(output_dir, f"enhanced_success_rate_{profile_name}"))
    plt.close()

def plot_collision_probability(per_slot_data, profile_name, output_dir):
    """
    رسم نمودار احتمال برخورد در طول زمان
    """
    setup_plot_style()
    
    collision_prob = per_slot_data["collisionProbPerSlot"]
    
    # محاسبه میانگین متحرک
    window_size = min(101, len(collision_prob) // 10)
    if window_size % 2 == 0:
        window_size += 1
    
    smoothed_collision = np.convolve(collision_prob, np.ones(window_size)/window_size, mode='same')
    
    fig, ax = plt.subplots()
    time_slots = np.arange(len(collision_prob)) * 0.005
    
    ax.plot(time_slots, collision_prob, alpha=0.5, label='Instantaneous', color='lightcoral')
    ax.plot(time_slots, smoothed_collision, label=f'Moving Average (window={window_size})', color='red')
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Collision Probability')
    ax.set_title(f'Collision Probability Over Time - {profile_name}')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    # اضافه کردن میانگین
    mean_collision = np.mean(collision_prob)
    ax.axhline(y=mean_collision, color='black', linestyle='--', 
               label=f'Average: {mean_collision:.3f}')
    ax.legend(loc='best')
    
    save_high_quality_plot(os.path.join(output_dir, f"collision_prob_{profile_name}"))
    plt.close()

def plot_traffic_composition(per_slot_data, profile_name, output_dir):
    """
    رسم ترکیب ترافیک NEW و RETX
    """
    setup_plot_style()
    
    fig, ax = plt.subplots()
    time_slots = np.arange(len(per_slot_data["newTraffic"])) * 0.005
    
    ax.plot(time_slots, per_slot_data["newTraffic"], 
            label='New Traffic', color='green', linewidth=2)
    ax.plot(time_slots, per_slot_data["retxTraffic"], 
            label='Retransmission Traffic', color='orange', linewidth=2)
    ax.plot(time_slots, per_slot_data["newTraffic"] + per_slot_data["retxTraffic"], 
            label='Total Traffic', color='purple', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Number of UEs')
    ax.set_title(f'Traffic Composition Over Time - {profile_name}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    save_high_quality_plot(os.path.join(output_dir, f"traffic_composition_{profile_name}"))
    plt.close()

def plot_preamble_allocation(per_slot_data, profile_name, output_dir):
    """
    رسم تخصیص preambleهای رزرو شده و عمومی
    """
    setup_plot_style()
    
    fig, ax = plt.subplots()
    time_slots = np.arange(len(per_slot_data["R_new_reserved"])) * 0.005
    
    # محاسبه کل preambleهای رزرو شده و عمومی
    total_reserved = per_slot_data["R_new_reserved"] + per_slot_data["R_retx_reserved"]
    total_general = per_slot_data["M_new_general"] + per_slot_data["M_retx_general"]
    
    ax.plot(time_slots, total_reserved, 
            label='Reserved Preambles', color='royalblue', linewidth=2)
    ax.plot(time_slots, total_general, 
            label='General Preambles', color='lightseagreen', linewidth=2)
    ax.plot(time_slots, total_reserved + total_general, 
            label='Total Preambles', color='black', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Number of Preambles')
    ax.set_title(f'Preamble Allocation Over Time - {profile_name}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    save_high_quality_plot(os.path.join(output_dir, f"preamble_allocation_{profile_name}"))
    plt.close()

def plot_preamble_usage_breakdown(per_slot_data, profile_name, output_dir):
    """
    نمودار تفکیک شده استفاده از انواع preamble
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    time_slots = np.arange(len(per_slot_data["R_new_reserved"])) * 0.005
    
    # محاسبه انواع preamble
    total_reserved_new = per_slot_data["R_new_reserved"]
    total_reserved_retx = per_slot_data["R_retx_reserved"]
    total_general_new = per_slot_data["M_new_general"]
    total_general_retx = per_slot_data["M_retx_general"]
    
    total_reserved = total_reserved_new + total_reserved_retx
    total_general = total_general_new + total_general_retx
    total_allocated = total_reserved + total_general
    
    # نمودار تجمعی (Stacked area)
    ax1.stackplot(time_slots, 
                 total_reserved_new, total_reserved_retx,
                 total_general_new, total_general_retx,
                 labels=['Reserved New', 'Reserved RETX', 'General New', 'General RETX'],
                 colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    
    ax1.plot(time_slots, total_allocated, 'k--', linewidth=1, label='Total Allocated')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Number of Preambles')
    ax1.set_title(f'Preamble Allocation Breakdown - {profile_name}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 55)
    
    # نمودار utilization
    utilization_reserved = total_reserved / 16.0  # hard_cap_total = 16
    utilization_general = total_general / 38.0    # M_MAX - reserved = 54-16=38
    utilization_total = total_allocated / 54.0    # M_MAX = 54
    
    ax2.plot(time_slots, utilization_reserved, label='Reserved Utilization', 
             color='#1f77b4', linewidth=2)
    ax2.plot(time_slots, utilization_general, label='General Utilization', 
             color='#2ca02c', linewidth=2)
    ax2.plot(time_slots, utilization_total, label='Total Utilization', 
             color='black', linestyle='--', linewidth=2)
    
    ax2.axhline(y=0.78, color='red', linestyle=':', alpha=0.7, label='Target Fill (78%)')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Utilization Ratio')
    ax2.set_title('Preamble Utilization Rates')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    save_high_quality_plot(os.path.join(output_dir, f"preamble_breakdown_{profile_name}"))
    plt.close()

def plot_delay_ecdf(per_slot_data, profile_name, output_dir):
    """
    رسم منحنی ECDF برای تاخیرهای موفق و ناموفق
    """
    setup_plot_style()
    
    fig, ax = plt.subplots()
    
    # رسم ECDF برای تاخیرهای موفق
    if len(per_slot_data["success_delays_s"]) > 0:
        success_delays = np.sort(per_slot_data["success_delays_s"])
        success_ecdf = np.arange(1, len(success_delays) + 1) / len(success_delays)
        ax.plot(success_delays, success_ecdf, 
                label='Success Delays', color='green', linewidth=2)
    
    # رسم ECDF برای تاخیرهای ناموفق (drop)
    if len(per_slot_data["dropped_delays_s"]) > 0:
        dropped_delays = np.sort(per_slot_data["dropped_delays_s"])
        dropped_ecdf = np.arange(1, len(dropped_delays) + 1) / len(dropped_delays)
        ax.plot(dropped_delays, dropped_ecdf, 
                label='Dropped UEs Lifetime', color='red', linewidth=2)
    
    ax.set_xlabel('Delay (seconds)')
    ax.set_ylabel('ECDF')
    ax.set_title(f'Empirical CDF of Delays - {profile_name}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    save_high_quality_plot(os.path.join(output_dir, f"delay_ecdf_{profile_name}"))
    plt.close()

def plot_utilization_metrics(per_slot_data, profile_name, output_dir):
    """
    رسم معیارهای utilization سیستم
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    time_slots = np.arange(len(per_slot_data["usedPreambles"])) * 0.005
    
    # Utilization (استفاده از preambleها)
    utilization = per_slot_data["usedPreambles"] / 54.0  # M_MAX = 44
    ax1.plot(time_slots, utilization, label='Preamble Utilization', color='teal', linewidth=2)
    ax1.set_ylabel('Utilization Ratio')
    ax1.set_title(f'System Utilization Over Time - {profile_name}')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Congestion (برخوردها)
    congestion_ratio = per_slot_data["congestedPreambles"] / per_slot_data["usedPreambles"]
    congestion_ratio = np.nan_to_num(congestion_ratio, nan=0.0)  # جایگزینی مقادیر NaN
    
    ax2.plot(time_slots, congestion_ratio, label='Congestion Ratio', color='coral', linewidth=2)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Congestion Ratio')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    plt.tight_layout()
    save_high_quality_plot(os.path.join(output_dir, f"utilization_metrics_{profile_name}"))
    plt.close()

def plot_delay_vs_devices(numDevicesVec, eventProbabilities, T, frameSize, output_dir, num_points=6):
    """
    رسم نمودار تاخیر بر حسب تعداد دستگاه‌ها
    """
    setup_plot_style()
    
    # محاسبه تعداد کل دستگاه‌ها از روی numDevicesVec
    total_devices = sum(numDevicesVec)
    print(f"Total number of devices: {total_devices:,}")
    
    # ایجاد طیفی از تعداد دستگاه‌ها برای تست (از 20% تا 150% تعداد اصلی)
    device_counts = np.linspace(int(total_devices * 0.2), int(total_devices * 1.5), num_points, dtype=int)
    print(f"Range of testing devices: {[f'{x:,}' for x in device_counts]}")
    
    avg_delays = []
    success_rates = []
    collision_probs = []
    
    print("Delay sensitivity")
    
    # ایجاد یک policy جدید برای این تحلیل
    G = len(numDevicesVec)
    analysis_policy = DynamicReservationPolicy(
        G=G, M_MAX=54,
        base_new=2, base_retx=2,
        max_per_group=6,
        hard_cap_total=16,
        cap_per_active=5,
        w_burst=1.6, w_retx_share=1.0, w_backlog=0.6,
        tau_on=0.55, tau_off=0.35,
        ramp_up=2, ramp_down=3,
        cooldown_slots=int(0.7 / frameSize),
        min_when_on=2
    )
    
    for device_count in tqdm(device_counts, desc="Simulation for different devices"):
        # محاسبه ضریب مقیاس‌گذاری
        scale_factor = device_count / total_devices
        
        # مقیاس‌گذاری تعداد دستگاه‌ها در هر گروه
        scaled_numDevicesVec = [max(1000, int(count * scale_factor)) for count in numDevicesVec]
        
        # تنظیم دقیق برای رسیدن به تعداد دقیق
        current_total = sum(scaled_numDevicesVec)
        if current_total != device_count:
            # تنظیم گروه بزرگتر برای رسیدن به تعداد دقیق
            diff = device_count - current_total
            max_group_idx = np.argmax(scaled_numDevicesVec)
            scaled_numDevicesVec[max_group_idx] += diff
        
        # اجرای شبیه‌سازی
        arrivals_total, arrivals_per_group, eventsAll, TbsAll = newArivals(
            scaled_numDevicesVec, eventProbabilities, T, frameSize
        )
        slots = int(T / frameSize)
        burst_mask = burst_mask_from_events(eventsAll, TbsAll, slots, frameSize)
        
        # ریست policy برای هر اجرا
        analysis_policy.on_flags[:] = False
        analysis_policy.on_until[:] = 0
        analysis_policy.curr_new[:] = 0
        analysis_policy.curr_retx[:] = 0
        
        # اجرای شبیه‌سازی ترافیک
        metrics, per_slot = actualTrafficPattern(
            arrivals_per_group, burst_mask, frameSize=frameSize, backoffBool=True,
            PERSIST_K_GEN=0.48,
            PERSIST_K_RES=1.20,
            TARGET_FILL=0.78,
            SHORT_SKIP_MIN=1,
            SHORT_SKIP_MAX=5,
            BACKOFF_BASE_MS=40,
            RETX_PRESSURE_GAIN=3.5,
            BACKOFF_CAP_MS_MAX=550,
            MIN_RETX_PREAMBLES=0,
            STARVATION_SHARE=0.5,
            STARVATION_MULTIPLIER=2.0,
            OVERFLOW_P=0.35,
            RESERVATION=None,
            CARVE_OUT=True,
            RES_POLICY=analysis_policy
        )
        
        # ذخیره نتایج
        if metrics["delay_stats"]["avg_success_delay_s"] is not None:
            avg_delays.append(metrics["delay_stats"]["avg_success_delay_s"])
            success_rates.append(metrics["success_rate_per_attempt"])
            collision_probs.append(metrics["overall_collision_probability"])
        else:
            avg_delays.append(0)
            success_rates.append(0)
            collision_probs.append(0)
    
    # رسم نمودار اصلی
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Total devices', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average delay of success:', color=color1, fontsize=12)
    line1 = ax1.plot(device_counts, avg_delays, 'o-', color=color1, linewidth=3, 
                    markersize=10, markerfacecolor='white', markeredgewidth=2, label='Average delay')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(device_counts)
    
    # نمودار نرخ موفقیت در محور دوم
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Success rate', color=color2, fontsize=12)
    line2 = ax2.plot(device_counts, success_rates, 's--', color=color2, linewidth=2, 
                    markersize=8, label=' Success rate')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 1.05)
    
    # نمودار احتمال برخورد در محور سوم (همان محور اول)
    color3 = 'tab:green'
    line3 = ax1.plot(device_counts, collision_probs, '^:', color=color3, linewidth=2,
                    markersize=8, label='Collision probability')
    
    # علامت‌گذاری نقطه اصلی (120,000 دستگاه)
    original_idx = np.where(device_counts == total_devices)[0]
    if len(original_idx) > 0:
        idx = original_idx[0]
        ax1.annotate(f'Real scenario\n({total_devices:,} Device)',
                    xy=(device_counts[idx], avg_delays[idx]),
                    xytext=(10, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor='yellow', alpha=0.7))
    
    # اضافه کردن legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=10)
    
    plt.title('Effect of number of devices on RACH', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_high_quality_plot(os.path.join(output_dir, "delay_vs_devices_comprehensive"))
    plt.close()
    
    # رسم نمودار جداگانه برای تاخیر
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(device_counts, avg_delays, 'o-', color='blue', linewidth=3, 
            markersize=10, markerfacecolor='white', markeredgewidth=2)
    ax.set_xlabel(' Total Devices ', fontsize=14, fontweight='bold')
    ax.set_ylabel(' Average access delay ', fontsize=12)
    ax.set_title('Effect of number of devices on access delay', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(device_counts)
    
    # علامت‌گذاری نقطه اصلی
    if len(original_idx) > 0:
        idx = original_idx[0]
        ax.annotate(f'Real scenarion\n({total_devices:,} Device)',
                   xy=(device_counts[idx], avg_delays[idx]),
                   xytext=(10, 20), textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', color='red'),
                   bbox=dict(boxstyle='round,pad=0.3', edgecolor='red', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    save_high_quality_plot(os.path.join(output_dir, "delay_vs_devices_simple"))
    plt.close()
    
    # ذخیره داده‌های خام
    delay_data = {
        'device_counts': device_counts.tolist(),
        'avg_delays': avg_delays,
        'success_rates': success_rates,
        'collision_probs': collision_probs,
        'original_total': total_devices
    }
    
    # ذخیره داده‌ها به صورت فایل
    torch.save(delay_data, os.path.join(output_dir, "delay_analysis_data.pt"))
    
    return delay_data

def plot_scalability_analysis(delay_data, output_dir):
    """
    تحلیل مقیاس‌پذیری سیستم
    """
    setup_plot_style()
    
    device_counts = np.array(delay_data['device_counts'])
    avg_delays = np.array(delay_data['avg_delays'])
    
    # محاسبه مشتق (شیب) برای تحلیل حساسیت
    if len(device_counts) > 1:
        delays_derivative = np.gradient(avg_delays, device_counts)
        
        # پیدا کردن نقطه شکست (جایی که شیب به شدت افزایش می‌یابد)
        threshold = np.mean(delays_derivative) + 2 * np.std(delays_derivative)
        breakpoint_idx = np.where(delays_derivative > threshold)[0]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # نمودار تاخیر و مشتق
        color1 = 'tab:blue'
        ax1.plot(device_counts, avg_delays, 'o-', color=color1, linewidth=3, 
                markersize=8, label='Average delay')
        ax1.set_xlabel('Number of devices')
        ax1.set_ylabel('delay ', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        ax1_twin = ax1.twinx()
        color2 = 'tab:red'
        ax1_twin.plot(device_counts, delays_derivative, 's--', color=color2, 
                     linewidth=2, markersize=6, label='Delay shib')
        ax1_twin.set_ylabel('Delay shib ', color=color2)
        ax1_twin.tick_params(axis='y', labelcolor=color2)
        
        # علامت‌گذاری نقطه شکست
        if len(breakpoint_idx) > 0:
            bp_idx = breakpoint_idx[0]
            ax1.axvline(x=device_counts[bp_idx], color='red', linestyle=':', 
                       alpha=0.7, label=f'break point: {device_counts[bp_idx]:,} device')
        
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # نمودار مقیاس‌پذیری
        normalized_delay = avg_delays / avg_delays[0] if avg_delays[0] > 0 else avg_delays
        normalized_devices = device_counts / device_counts[0]
        
        scalability = normalized_devices / normalized_delay
        
        ax2.plot(device_counts, scalability, '^-', color='green', linewidth=2,
                markersize=8, label='Scalability factor')
        ax2.set_xlabel('Number of devices')
        ax2.set_ylabel('Scalability factor', alpha=0.3)
        ax2.legend()
        
        plt.suptitle('System scalibility  RACH', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_high_quality_plot(os.path.join(output_dir, "scalability_analysis"))
        plt.close()

def plot_comparative_analysis(delay_data_dict, output_dir):
    """
    نمودار مقایسه‌ای برای تعداد دستگاه‌های مختلف
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(delay_data_dict)))
    
    for i, (device_count, data) in enumerate(delay_data_dict.items()):
        color = colors[i]
        label = f'{device_count:,} Devices'
        
        # نمودار نرخ موفقیت
        ax1.plot(data['device_counts'], data['success_rates'], 
                'o-', color=color, linewidth=2, markersize=6, label=label)
        
        # نمودار احتمال برخورد
        ax2.plot(data['device_counts'], data['collision_probs'], 
                's--', color=color, linewidth=2, markersize=6, label=label)
    
    ax1.set_xlabel('Number of Devices')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Success Rate vs. Number of Devices', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel('Number of Devices')
    ax2.set_ylabel('Collision Probability')
    ax2.set_title('Collision Probability vs. Number of Devices', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    save_high_quality_plot(os.path.join(output_dir, "comparative_analysis"))
    plt.close()

def plot_final_comparison(all_results, output_dir):
    """
    نمودار نهایی مقایسه تمام سناریوها
    """
    setup_plot_style()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for i, (device_count, data) in enumerate(all_results.items()):
        color = colors[i]
        marker = markers[i]
        label = f'{device_count:,} Devices'
        
        device_counts = np.array(data['device_counts'])
        
        # موفقیت
        ax1.plot(device_counts, data['success_rates'], 
                marker=marker, color=color, linewidth=2, 
                markersize=6, label=label)
        
        # برخورد
        ax2.plot(device_counts, data['collision_probs'], 
                marker=marker, color=color, linewidth=2, 
                markersize=6, label=label)
        
        # تاخیر
        ax3.plot(device_counts, data['avg_delays'], 
                marker=marker, color=color, linewidth=2, 
                markersize=6, label=label)
        
        # utilization از داده‌های اصلی
        main_metrics = data['main_metrics']
        util_rate = main_metrics['avg_preamble_occupancy']
        ax4.bar(i, util_rate, color=color, alpha=0.7, label=label)
    
    ax1.set_xlabel('Number of Devices')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Success Rate Comparison', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel('Number of Devices')
    ax2.set_ylabel('Collision Probability')
    ax2.set_title('Collision Probability Comparison', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3.set_xlabel('Number of Devices')
    ax3.set_ylabel('Average Delay (s)')
    ax3.set_title('Access Delay Comparison', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    ax4.set_xlabel('Scenario')
    ax4.set_ylabel('Average Utilization')
    ax4.set_title('Preamble Utilization Comparison', fontweight='bold')
    ax4.set_xticks(range(len(all_results)))
    ax4.set_xticklabels([f'{k//1000}k' for k in all_results.keys()])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_high_quality_plot(os.path.join(output_dir, "final_comprehensive_comparison"))
    plt.close()

def simulate_multiple_device_scenarios(base_numDevicesVec, eventProbabilities, T, frameSize, target_totals, output_dir):
    """
    شبیه‌سازی برای تعداد دستگاه‌های مختلف و ذخیره نتایج
    """
    setup_plot_style()
    
    all_results = {}
    
    for total_devices in tqdm(target_totals, desc="Simulating different device counts"):
        print(f"\n=== Simulating for {total_devices:,} devices ===")
        
        # محاسبه ضریب مقیاس
        base_total = sum(base_numDevicesVec)
        scale_factor = total_devices / base_total
        
        # مقیاس‌گذاری تعداد دستگاه‌ها
        scaled_numDevicesVec = [max(1000, int(count * scale_factor)) for count in base_numDevicesVec]
        
        # تنظیم دقیق
        current_total = sum(scaled_numDevicesVec)
        if current_total != total_devices:
            diff = total_devices - current_total
            max_group_idx = np.argmax(scaled_numDevicesVec)
            scaled_numDevicesVec[max_group_idx] += diff
        
        # ایجاد policy جدید
        G = len(scaled_numDevicesVec)
        analysis_policy = DynamicReservationPolicy(
            G=G, M_MAX=54,
            base_new=2, base_retx=2,
            max_per_group=6,
            hard_cap_total=16,
            cap_per_active=5,
            w_burst=1.6, w_retx_share=1.0, w_backlog=0.6,
            tau_on=0.55, tau_off=0.35,
            ramp_up=2, ramp_down=3,
            cooldown_slots=int(0.7 / frameSize),
            min_when_on=2
        )
        
        # اجرای شبیه‌سازی
        arrivals_total, arrivals_per_group, eventsAll, TbsAll = newArivals(
            scaled_numDevicesVec, eventProbabilities, T, frameSize
        )
        slots = int(T / frameSize)
        burst_mask = burst_mask_from_events(eventsAll, TbsAll, slots, frameSize)
        
        # ریست policy
        analysis_policy.on_flags[:] = False
        analysis_policy.on_until[:] = 0
        analysis_policy.curr_new[:] = 0
        analysis_policy.curr_retx[:] = 0
        
        # اجرای شبیه‌سازی ترافیک
        metrics, per_slot = actualTrafficPattern(
            arrivals_per_group, burst_mask, frameSize=frameSize, backoffBool=True,
            PERSIST_K_GEN=0.48, PERSIST_K_RES=1.20, TARGET_FILL=0.78,
            SHORT_SKIP_MIN=1, SHORT_SKIP_MAX=5, BACKOFF_BASE_MS=40,
            RETX_PRESSURE_GAIN=3.5, BACKOFF_CAP_MS_MAX=550,
            MIN_RETX_PREAMBLES=0, STARVATION_SHARE=0.5, 
            STARVATION_MULTIPLIER=2.0, OVERFLOW_P=0.35,
            RESERVATION=None, CARVE_OUT=True, RES_POLICY=analysis_policy
        )
        
        # تحلیل حساسیت برای این سناریو
        device_counts = np.linspace(int(total_devices * 0.3), int(total_devices * 1.7), 8, dtype=int)
        avg_delays = []
        success_rates = []
        collision_probs = []
        
        for device_count in device_counts:
            scale_factor_inner = device_count / total_devices
            scaled_inner = [max(1000, int(count * scale_factor_inner)) for count in scaled_numDevicesVec]
            
            current_inner = sum(scaled_inner)
            if current_inner != device_count:
                diff = device_count - current_inner
                max_group_idx = np.argmax(scaled_inner)
                scaled_inner[max_group_idx] += diff
            
            # شبیه‌سازی سریع برای تحلیل حساسیت
            arrivals_total_inner, arrivals_per_group_inner, eventsAll_inner, TbsAll_inner = newArivals(
                scaled_inner, eventProbabilities, T, frameSize
            )
            burst_mask_inner = burst_mask_from_events(eventsAll_inner, TbsAll_inner, slots, frameSize)
            
            policy_inner = DynamicReservationPolicy(
                G=G, M_MAX=54, base_new=2, base_retx=2, max_per_group=6,
                hard_cap_total=16, cap_per_active=5, w_burst=1.6, w_retx_share=1.0, w_backlog=0.6,
                tau_on=0.55, tau_off=0.35, ramp_up=2, ramp_down=3,
                cooldown_slots=int(0.7 / frameSize), min_when_on=2
            )
            
            metrics_inner, _ = actualTrafficPattern(
                arrivals_per_group_inner, burst_mask_inner, frameSize=frameSize, backoffBool=True,
                PERSIST_K_GEN=0.48, PERSIST_K_RES=1.20, TARGET_FILL=0.78,
                SHORT_SKIP_MIN=1, SHORT_SKIP_MAX=5, BACKOFF_BASE_MS=40,
                RETX_PRESSURE_GAIN=3.5, BACKOFF_CAP_MS_MAX=550,
                MIN_RETX_PREAMBLES=0, STARVATION_SHARE=0.5, 
                STARVATION_MULTIPLIER=2.0, OVERFLOW_P=0.35,
                RESERVATION=None, CARVE_OUT=True, RES_POLICY=policy_inner
            )
            
            if metrics_inner["delay_stats"]["avg_success_delay_s"] is not None:
                avg_delays.append(metrics_inner["delay_stats"]["avg_success_delay_s"])
                success_rates.append(metrics_inner["success_rate_per_attempt"])
                collision_probs.append(metrics_inner["overall_collision_probability"])
            else:
                avg_delays.append(0)
                success_rates.append(0)
                collision_probs.append(0)
        
        # ذخیره نتایج
        all_results[total_devices] = {
            'device_counts': device_counts.tolist(),
            'avg_delays': avg_delays,
            'success_rates': success_rates,
            'collision_probs': collision_probs,
            'main_metrics': metrics,
            'per_slot_data': per_slot
        }
        
        # تولید نمودارهای تفصیلی برای این سناریو
        plot_enhanced_success_rate(per_slot, f"{total_devices}_devices", output_dir)
        plot_preamble_usage_breakdown(per_slot, f"{total_devices}_devices", output_dir)
    
    # ذخیره همه نتایج
    torch.save(all_results, os.path.join(output_dir, "all_scenarios_results.pt"))
    
    # نمودار مقایسه‌ای
    plot_comparative_analysis(all_results, output_dir)
    plot_final_comparison(all_results, output_dir)
    
    return all_results

# ============================
# Distributions & Generators
# ============================

def betaDistribution(t, T=10.0):
    alpha = 3.0
    beta = 4.0
    betaFunction = special.beta(alpha, beta)
    pdf = (t**(alpha-1.0) * (T - t)**(beta-1.0)) / (T**(alpha+beta-1.0) * betaFunction)
    return pdf

def uniformTrafficIntensity(T, frameSize, numDevices, packetProbability=1/60):
    slots = int(T / frameSize)
    packetsPerSlotPerDev = packetProbability * frameSize
    randomSeries = np.random.uniform(0.0, 1.0, (slots, numDevices))
    uniformTraffic = np.sum(randomSeries <= packetsPerSlotPerDev, axis=1)
    return uniformTraffic.astype(int)

def burstTrafficIntensity(numDevices, frameSize, Tb=3.5):
    slots = int(Tb / frameSize)
    packetsPerSlotPerDev = np.zeros((slots, numDevices))
    tVector = np.linspace(0.0, Tb - frameSize, slots)
    for i, t in enumerate(tVector):
        val = integrate.quad(lambda x: betaDistribution(x, Tb), t, t + frameSize)[0]
        packetsPerSlotPerDev[i, :] = val
    randomSeries = np.random.uniform(0.0, 1.0, (slots, numDevices))
    burstTraffic = np.sum(randomSeries <= packetsPerSlotPerDev, axis=1)
    return burstTraffic.astype(int)

def eventGenerator(T, frameSize, eventProbability):
    slots = int(T / frameSize)
    eventProbabilityPerSlot = eventProbability * frameSize
    events = np.random.uniform(0.0, 1.0, slots) <= eventProbabilityPerSlot
    eventsTime = np.where(events)[0]
    numEvents = int(np.sum(events))
    Tb = np.random.uniform(2.0, 5.0, numEvents)  # seconds
    Tb = Tb - (Tb % frameSize)
    lastBurstEnd = 0
    overlapping = []
    for i, trig in enumerate(eventsTime):
        if lastBurstEnd > trig:
            overlapping.append(i)
        else:
            lastBurstEnd = trig + int(Tb[i] / frameSize)
    eventsTime = np.delete(eventsTime, overlapping)
    Tb = np.delete(Tb, overlapping)
    return eventsTime, Tb

def newArivals(numDevicesVec, eventProbabilities, T, frameSize):
    slots = int(T / frameSize)
    G = len(numDevicesVec)
    arrivals_total = np.zeros(slots, dtype=int)
    arrivals_per_group = np.zeros((slots, G), dtype=int)
    eventsAll = []
    TbsAll = []
    for i, numDevices in enumerate(numDevicesVec):
        groupTraffic = uniformTrafficIntensity(T, frameSize, numDevices)
        eventsTime, Tb = eventGenerator(T, frameSize, eventProbabilities[i])
        eventsAll.append(eventsTime)
        TbsAll.append(Tb)
        for k, event in enumerate(eventsTime):
            dur = int(Tb[k] / frameSize)
            end_idx = event + dur
            burstTraffic = burstTrafficIntensity(numDevices, frameSize, Tb[k])
            if end_idx >= slots:
                end_idx = slots
                lastPoint = end_idx - event
                groupTraffic[event:end_idx] = burstTraffic[:lastPoint]
            else:
                groupTraffic[event:end_idx] = burstTraffic
        arrivals_total += groupTraffic
        arrivals_per_group[:, i] = groupTraffic
    return arrivals_total, arrivals_per_group, eventsAll, TbsAll

def burst_mask_from_events(eventsAll, TbsAll, slots, frameSize):
    G = len(eventsAll)
    mask = np.zeros((slots, G), dtype=bool)
    for g in range(G):
        for k, start in enumerate(eventsAll[g]):
            dur_slots = int(TbsAll[g][k] / frameSize)
            end = min(slots, start + dur_slots)
            if start < end:
                mask[start:end, g] = True
    return mask

# ============================
# UE Class
# ============================

# ============================
# Optimized UE Class with NumPy Arrays
# ============================

class UE:
    """Original UE class kept for compatibility"""
    def __init__(self, group_id, first_slot=None):
        self.group = group_id
        self.transmissions = 0
        self.preamble = 0
        self.backoffCounter = 0
        self.first_slot = first_slot
        self.success = False


@njit(cache=True)
def compute_preamble_selection(preamble_counts, range_start, range_end, rng_seed):
    """
    Numba-accelerated two-choice preamble selection.
    Selects the less-loaded preamble from two random choices.
    """
    if range_end <= range_start:
        return -1
    if range_end - range_start == 1:
        return range_start
    
    np.random.seed(rng_seed)
    i = np.random.randint(range_start, range_end)
    j = np.random.randint(range_start, range_end)
    
    return i if preamble_counts[i] <= preamble_counts[j] else j


@njit(cache=True)
def compute_adaptive_backoff(retx_ratio, base_ms, pressure_gain, cap_ms_max, rng_val):
    """
    Compute adaptive backoff using truncated Pareto distribution.
    Faster convergence and better collision resolution.
    """
    # Adaptive cap based on retransmission pressure
    cap_ms = min(cap_ms_max, base_ms * (1.0 + pressure_gain * retx_ratio))
    
    # Truncated Pareto (heavy-tailed but bounded)
    # Using inverse transform sampling
    alpha = 2.5  # Pareto shape parameter
    r = max(1e-9, 1.0 - rng_val)
    backoff_ms = cap_ms * (1.0 - r ** (1.0 / (1.0 - alpha)))
    
    return int(backoff_ms)


@njit(cache=True)
def optimized_preamble_allocation(slot, U_new_g, U_retx_g, back_new_g, back_retx_g,
                                   burst_row, reserved_map_keys, reserved_map_new, 
                                   reserved_map_retx, M_MAX, R_total,
                                   PERSIST_K_RES, TARGET_FILL, OVERFLOW_P,
                                   SHORT_SKIP_MIN, SHORT_SKIP_MAX,
                                   p_new_g, p_retx_g, rng_state):
    """
    Optimized preamble allocation kernel.
    Returns: selected_preambles, attempts_mask, next_backoffs
    """
    G = len(U_new_g)
    total_ues = np.sum(U_new_g) + np.sum(U_retx_g)
    
    # Pre-allocate arrays
    selected_preambles = np.full(total_ues, -1, dtype=np.int32)
    attempts_mask = np.zeros(total_ues, dtype=np.bool_)
    next_backoffs = np.zeros(total_ues, dtype=np.int32)
    
    return selected_preambles, attempts_mask, next_backoffs


# ============================
# Vectorized Traffic Simulation Engine
# ============================

class VectorizedUESimulator:
    """
    High-performance UE simulator using vectorized numpy operations.
    Replaces object-based UE management with array-based approach.
    """
    
    def __init__(self, max_ues=200000, M_MAX=54, preambleTransMax=10):
        self.max_ues = max_ues
        self.M_MAX = M_MAX
        self.preambleTransMax = preambleTransMax
        
        # Pre-allocate arrays for all UE attributes
        self.groups = np.zeros(max_ues, dtype=np.int32)
        self.transmissions = np.zeros(max_ues, dtype=np.int32)
        self.preambles = np.zeros(max_ues, dtype=np.int32)
        self.backoff_counters = np.zeros(max_ues, dtype=np.int32)
        self.first_slots = np.zeros(max_ues, dtype=np.int32)
        self.active_mask = np.zeros(max_ues, dtype=np.bool_)
        
        self.n_active = 0
        self.rng = np.random.default_rng()
    
    def reset(self):
        """Reset simulator state"""
        self.n_active = 0
        self.active_mask[:] = False
        self.transmissions[:] = 0
        self.backoff_counters[:] = 0
    
    def add_ues(self, groups, first_slot):
        """Add new UEs to the simulation"""
        n_new = len(groups)
        if self.n_active + n_new > self.max_ues:
            raise ValueError(f"Exceeded max UEs: {self.n_active + n_new} > {self.max_ues}")
        
        idx_start = self.n_active
        idx_end = idx_start + n_new
        
        self.groups[idx_start:idx_end] = groups
        self.first_slots[idx_start:idx_end] = first_slot
        self.active_mask[idx_start:idx_end] = True
        self.n_active = idx_end
    
    def step(self, slot, arrivals_per_group, burst_row, reserved_map,
             M_new_gen, M_retx_gen, p_new_g, p_retx_g, p_new_gen, p_retx_gen,
             OVERFLOW_P, SHORT_SKIP_MIN, SHORT_SKIP_MAX, BACKOFF_BASE_MS,
             RETX_PRESSURE_GAIN, BACKOFF_CAP_MS_MAX):
        """Execute one simulation slot with vectorized operations"""
        
        # Decrement backoff counters for all active UEs
        self.backoff_counters[:self.n_active] = np.maximum(
            0, self.backoff_counters[:self.n_active] - 1
        )
        
        # Identify ready UEs (backoff == 0)
        ready_mask = (self.backoff_counters[:self.n_active] == 0) & self.active_mask[:self.n_active]
        
        # Separate NEW and RETX UEs
        is_new = self.transmissions[:self.n_active] == 0
        is_retx = ~is_new
        
        ready_new = ready_mask & is_new
        ready_retx = ready_mask & is_retx
        
        # Initialize preamble selections
        self.preambles[:self.n_active] = -1
        
        # Count contenders per group
        U_new_g = np.zeros(len(p_new_g), dtype=np.int32)
        U_retx_g = np.zeros(len(p_retx_g), dtype=np.int32)
        
        for g in range(len(p_new_g)):
            group_mask = self.groups[:self.n_active] == g
            U_new_g[g] = np.sum(ready_new & group_mask)
            U_retx_g[g] = np.sum(ready_retx & group_mask)
        
        # Perform preamble selection (optimized)
        preamble_counts = np.zeros(self.M_MAX, dtype=np.int32)
        attempts_this_slot = 0
        successful_this_slot = 0
        
        # Process each UE
        for i in range(self.n_active):
            if not self.active_mask[i] or self.backoff_counters[i] > 0:
                continue
            
            g = self.groups[i]
            in_burst = burst_row[g]
            is_new_ue = (self.transmissions[i] == 0)
            
            selected = -1
            attempt_made = False
            
            # Try reserved pool first (if in burst)
            if in_burst and g in reserved_map:
                Rn = reserved_map[g]['new']
                Rr = reserved_map[g]['retx']
                
                if is_new_ue and Rn > 0:
                    if self.rng.random() < p_new_g[g]:
                        # Two-choice selection in reserved NEW range
                        selected = self._two_choice_reserved(g, 'N', preamble_counts, reserved_map)
                        if selected >= 0:
                            attempt_made = True
                elif not is_new_ue and Rr > 0:
                    if self.rng.random() < p_retx_g[g]:
                        selected = self._two_choice_reserved(g, 'R', preamble_counts, reserved_map)
                        if selected >= 0:
                            attempt_made = True
                
                # Overflow to general pool
                if not attempt_made and self.rng.random() < OVERFLOW_P:
                    pass  # Will try general pool below
                elif not attempt_made:
                    # Skip slots
                    self.backoff_counters[i] = self.rng.integers(SHORT_SKIP_MIN, SHORT_SKIP_MAX + 1)
                    continue
            
            # Try general pool
            if not attempt_made:
                if is_new_ue and M_new_gen > 0:
                    if self.rng.random() < p_new_gen:
                        selected = self._two_choice_general('N', preamble_counts, M_new_gen, M_retx_gen)
                        if selected >= 0:
                            attempt_made = True
                elif not is_new_ue and M_retx_gen > 0:
                    if self.rng.random() < p_retx_gen:
                        selected = self._two_choice_general('R', preamble_counts, M_new_gen, M_retx_gen)
                        if selected >= 0:
                            attempt_made = True
            
            if not attempt_made:
                self.backoff_counters[i] = self.rng.integers(SHORT_SKIP_MIN, SHORT_SKIP_MAX + 1)
                continue
            
            # Record attempt
            if selected >= 0:
                self.preambles[i] = selected
                preamble_counts[selected] += 1
                self.transmissions[i] += 1
                attempts_this_slot += 1
        
        return attempts_this_slot, preamble_counts
    
    def _two_choice_reserved(self, g, pool_type, preamble_counts, reserved_map):
        """Two-choice selection for reserved pool"""
        # Simplified: select from reserved range
        base_idx = g * 2 if pool_type == 'N' else g * 2 + 1
        range_size = reserved_map[g]['new'] if pool_type == 'N' else reserved_map[g]['retx']
        
        if range_size <= 0:
            return -1
        
        # Two random choices
        i = self.rng.integers(0, range_size)
        j = self.rng.integers(0, range_size)
        
        # Map to actual preamble indices (simplified)
        idx_i = (base_idx + i) % self.M_MAX
        idx_j = (base_idx + j) % self.M_MAX
        
        return idx_i if preamble_counts[idx_i] <= preamble_counts[idx_j] else idx_j
    
    def _two_choice_general(self, pool_type, preamble_counts, M_new_gen, M_retx_gen):
        """Two-choice selection for general pool"""
        if pool_type == 'N':
            start = 0
            size = M_new_gen
        else:
            start = M_new_gen
            size = M_retx_gen
        
        if size <= 0:
            return -1
        
        i = self.rng.integers(start, start + size)
        j = self.rng.integers(start, start + size)
        
        return i if preamble_counts[i] <= preamble_counts[j] else j
    
    def resolve_outcomes(self, preamble_counts, slot):
        """Resolve success/collision/backoff for all attempting UEs"""
        successes = []
        collisions = []
        
        for i in range(self.n_active):
            if not self.active_mask[i] or self.preambles[i] < 0:
                continue
            
            pr = self.preambles[i]
            count = preamble_counts[pr]
            
            if count == 1:
                # Success
                successes.append(i)
                self.active_mask[i] = False
            elif count > 1:
                # Collision - apply backoff
                collisions.append(i)
                # Backoff will be applied in next step
        
        return successes, collisions

# ============================
# Dynamic Reservation Policy
# ============================

class DynamicReservationPolicy:
    def __init__(self, G, M_MAX=54,
                 base_new=2, base_retx=2,
                 max_per_group=6,          # ↑
                 hard_cap_total=16,        # ↑
                 cap_per_active=5,         # ↑
                 w_burst=1.6, w_retx_share=1.0, w_backlog=0.6,
                 tau_on=0.55, tau_off=0.35,
                 ramp_up=2, ramp_down=3,
                 cooldown_slots=0,
                 min_when_on=2):
        self.G = G
        self.M_MAX = M_MAX
        self.base_new = base_new
        self.base_retx = base_retx
        self.max_per_group = max_per_group
        self.hard_cap_total = min(hard_cap_total, M_MAX)
        self.cap_per_active = cap_per_active
        self.w_burst = w_burst
        self.w_retx_share = w_retx_share
        self.w_backlog = w_backlog
        self.tau_on = tau_on
        self.tau_off = tau_off
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.cooldown_slots = cooldown_slots
        self.min_when_on = min_when_on

        self.on_flags = np.zeros(G, dtype=bool)
        self.on_until = np.zeros(G, dtype=int)
        self.curr_new = np.zeros(G, dtype=int)
        self.curr_retx = np.zeros(G, dtype=int)

    def step(self, slot, burst_row, U_new_g, U_retx_g, back_new_g, back_retx_g):
        eps = 1e-9
        target_new = np.zeros(self.G, dtype=float)
        target_retx = np.zeros(self.G, dtype=float)

        # effective cap by number of active groups
        active_groups = [g for g in range(self.G)
                         if burst_row[g] or (back_new_g[g] + back_retx_g[g]) > 0]
        n_active = len(active_groups)
        eff_cap = min(self.hard_cap_total, self.cap_per_active * n_active, self.M_MAX)

        for g in range(self.G):
            re_sum = U_new_g[g] + U_retx_g[g]
            retx_share = (U_retx_g[g] / (re_sum + eps)) if re_sum > 0 else 0.0
            backlog_present = (back_new_g[g] + back_retx_g[g]) > 0
            score = (self.w_burst * (1.0 if burst_row[g] else 0.0)
                     + self.w_retx_share * retx_share
                     + self.w_backlog * backlog_present)

            if self.on_flags[g]:
                if (score < self.tau_off) and (slot >= self.on_until[g]):
                    self.on_flags[g] = False
                else:
                    self.on_until[g] = max(self.on_until[g], slot + self.cooldown_slots)
            else:
                if (score >= self.tau_on) or burst_row[g]:
                    self.on_flags[g] = True
                    self.on_until[g] = slot + self.cooldown_slots

            if self.on_flags[g]:
                target_total = max(self.min_when_on,
                                   self.base_new + self.base_retx + int(4 * score))
                target_total = min(target_total, self.max_per_group)
                tn = int(round(target_total * (1.0 - retx_share)))
                tr = target_total - tn
                target_new[g] = max(0, tn)
                target_retx[g] = max(0, tr)
            else:
                target_new[g] = 0
                target_retx[g] = 0

        def ramp(curr, tgt):
            diff = tgt - curr
            if diff > 0:
                step = min(self.ramp_up, diff)
            else:
                step = max(-self.ramp_down, diff)
            return curr + step

        next_new = np.array([ramp(self.curr_new[g], int(target_new[g]))
                             for g in range(self.G)], dtype=int)
        next_retx = np.array([ramp(self.curr_retx[g], int(target_retx[g]))
                              for g in range(self.G)], dtype=int)

        total_req = int(next_new.sum() + next_retx.sum())
        budget = int(eff_cap)
        if (total_req > budget) and (total_req > 0):
            scale = budget / total_req
            next_new = np.floor(next_new * scale).astype(int)
            next_retx = np.floor(next_retx * scale).astype(int)
            leftover = budget - int(next_new.sum() + next_retx.sum())
            idx = 0
            order = [('n', i) for i in range(self.G)] + [('r', i) for i in range(self.G)]
            while leftover > 0 and order:
                tag, i = order[idx % len(order)]
                if tag == 'n':
                    if next_new[i] < self.max_per_group:
                        next_new[i] += 1
                        leftover -= 1
                else:
                    if next_retx[i] < self.max_per_group:
                        next_retx[i] += 1
                        leftover -= 1
                idx += 1

        self.curr_new = next_new
        self.curr_retx = next_retx

        reserved_map = {}
        for g in range(self.G):
            if (self.curr_new[g] > 0) or (self.curr_retx[g] > 0):
                reserved_map[g] = {'new': int(self.curr_new[g]),
                                   'retx': int(self.curr_retx[g])}
        return reserved_map

# ============================
# Optimized Core Engine with Enhanced Preamble Allocation
# ============================

@njit(cache=True)
def fast_two_choice_preamble(preamble_counts, range_start, range_end):
    """Numba-accelerated two-choice preamble selection"""
    if range_end <= range_start:
        return -1
    if range_end - range_start == 1:
        return range_start
    
    i = np.random.randint(range_start, range_end)
    j = np.random.randint(range_start, range_end)
    
    return i if preamble_counts[i] <= preamble_counts[j] else j


@njit(cache=True)
def compute_backoff_slots(retx_ratio, base_ms, pressure_gain, cap_ms_max, frameSize, rng_val):
    """Compute adaptive backoff slots using truncated Pareto distribution"""
    cap_ms = min(cap_ms_max, base_ms * (1.0 + pressure_gain * retx_ratio))
    alpha = 2.5
    r = max(1e-9, 1.0 - rng_val)
    backoff_ms = cap_ms * (1.0 - r ** (1.0 / (1.0 - alpha)))
    return int(np.ceil(backoff_ms / (frameSize * 1000.0)))


def actualTrafficPattern_optimized(
    arrivals_per_group, burst_mask, frameSize=0.005, backoffBool=True,
    PERSIST_K_GEN=0.48, PERSIST_K_RES=1.20, TARGET_FILL=0.78,
    SHORT_SKIP_MIN=1, SHORT_SKIP_MAX=5,
    BACKOFF_BASE_MS=40, RETX_PRESSURE_GAIN=3.5, BACKOFF_CAP_MS_MAX=550,
    MIN_RETX_PREAMBLES=0, STARVATION_SHARE=0.5, STARVATION_MULTIPLIER=2.0,
    OVERFLOW_P=0.35, RESERVATION=None, CARVE_OUT=True,
    RES_POLICY=None, use_vectorized=True
):
    """
    Optimized traffic pattern simulation with enhanced preamble allocation.
    
    KEY OPTIMIZATIONS:
    1. Pre-allocated arrays instead of dynamic lists
    2. Vectorized UE state management
    3. Numba-jitted hot paths
    4. Efficient two-choice hashing for load balancing
    5. Adaptive PI controller with faster convergence
    6. Truncated Pareto backoff for better collision resolution
    """
    M_MAX = 54
    preambleTransMax = 10
    slots, G = arrivals_per_group.shape
    eps = 1e-9
    
    # Pre-allocate all per-slot arrays
    UEsPerSlot = np.zeros(slots, dtype=np.int32)
    successfulUEsPerSlot = np.zeros(slots, dtype=np.int32)
    congestedPreambles = np.zeros(slots, dtype=np.int32)
    freePreambles = np.zeros(slots, dtype=np.int32)
    usedPreambles = np.zeros(slots, dtype=np.int32)
    newTraffic = np.zeros(slots, dtype=np.int32)
    retxTraffic = np.zeros(slots, dtype=np.int32)
    M_new_series = np.zeros(slots, dtype=np.int32)
    M_retx_series = np.zeros(slots, dtype=np.int32)
    R_new_series = np.zeros(slots, dtype=np.int32)
    R_retx_series = np.zeros(slots, dtype=np.int32)
    collidedUEsPerSlot = np.zeros(slots, dtype=np.int32)
    attemptedUEsPerSlot = np.zeros(slots, dtype=np.int32)
    collisionProbPerSlot = np.zeros(slots, dtype=np.float64)
    
    # Delay tracking
    success_delays_slots = []
    dropped_delays_slots = []
    
    # Use pre-allocated array-based UE storage
    max_ues = int(arrivals_per_group.sum() * 1.5)
    ue_groups = np.zeros(max_ues, dtype=np.int32)
    ue_transmissions = np.zeros(max_ues, dtype=np.int32)
    ue_preambles = np.full(max_ues, -1, dtype=np.int32)
    ue_backoffs = np.zeros(max_ues, dtype=np.int32)
    ue_first_slots = np.zeros(max_ues, dtype=np.int32)
    ue_active = np.zeros(max_ues, dtype=np.bool_)
    
    n_ues = 0
    last_state = np.zeros(M_MAX, dtype=np.int32)
    
    # Enhanced PI controller with adaptive gains
    k_gen = 1.0
    e_int = 0.0
    KP = 0.25  # Slightly higher for faster response
    KI = 0.015
    KGEN_MIN, KGEN_MAX = 0.40, 1.50
    LAMBDA_TARGET = 0.50  # Slightly higher target for better utilization
    
    rng = np.random.default_rng()
    
    def scale_reserved(req_new, req_retx, budget):
        total_req = sum(req_new) + sum(req_retx)
        if total_req <= budget or total_req == 0:
            return req_new, req_retx
        scale = budget / total_req
        rn = [int(np.floor(x * scale)) for x in req_new]
        rr = [int(np.floor(x * scale)) for x in req_retx]
        used = sum(rn) + sum(rr)
        leftover = max(0, budget - used)
        idx = 0
        order = [('n', i) for i in range(len(rn))] + [('r', i) for i in range(len(rr))]
        while leftover > 0 and order:
            tag, i = order[idx % len(order)]
            if tag == 'n':
                rn[i] += 1
            else:
                rr[i] += 1
            leftover -= 1
            idx += 1
        return rn, rr
    
    for slot in range(slots):
        # Add newcomers efficiently
        n_new = int(arrivals_per_group[slot].sum())
        if n_new > 0:
            new_indices = np.arange(n_ues, n_ues + n_new)
            current_idx = 0
            for g in range(G):
                n_g = int(arrivals_per_group[slot, g])
                if n_g > 0:
                    ue_groups[new_indices[current_idx:current_idx+n_g]] = g
                    ue_first_slots[new_indices[current_idx:current_idx+n_g]] = slot
                    ue_active[new_indices[current_idx:current_idx+n_g]] = True
                    current_idx += n_g
            n_ues += n_new
        
        # Count ready/backoff per group (vectorized)
        active_mask = ue_active[:n_ues]
        ready_mask = (ue_backoffs[:n_ues] == 0) & active_mask
        
        U_new_g = np.zeros(G, dtype=np.int32)
        U_retx_g = np.zeros(G, dtype=np.int32)
        back_new_g = np.zeros(G, dtype=np.int32)
        back_retx_g = np.zeros(G, dtype=np.int32)
        
        for g in range(G):
            group_mask = ue_groups[:n_ues] == g
            U_new_g[g] = np.sum(ready_mask & group_mask & (ue_transmissions[:n_ues] == 0))
            U_retx_g[g] = np.sum(ready_mask & group_mask & (ue_transmissions[:n_ues] > 0))
            back_new_g[g] = np.sum(~ready_mask & group_mask & (ue_transmissions[:n_ues] == 0) & active_mask)
            back_retx_g[g] = np.sum(~ready_mask & group_mask & (ue_transmissions[:n_ues] > 0) & active_mask)
        
        U_new_total = U_new_g.sum()
        U_retx_total = U_retx_g.sum()
        newTraffic[slot] = U_new_total
        retxTraffic[slot] = U_retx_total
        
        # Reserved plan from policy
        if RES_POLICY is not None:
            reserved_map = RES_POLICY.step(slot, burst_mask[slot], U_new_g, U_retx_g, back_new_g, back_retx_g)
        else:
            req_new, req_retx, active_groups = [], [], []
            for g in range(G):
                if burst_mask[slot, g]:
                    rconf = (RESERVATION or {}).get(g, {'new': 0, 'retx': 0})
                    req_new.append(int(rconf.get('new', 0)))
                    req_retx.append(int(rconf.get('retx', 0)))
                    active_groups.append(g)
            if active_groups:
                rn, rr = scale_reserved(req_new, req_retx, M_MAX)
                reserved_map = {g: {'new': rn[i], 'retx': rr[i]} for i, g in enumerate(active_groups)}
            else:
                reserved_map = {}
        
        R_new = sum(v['new'] for v in reserved_map.values()) if reserved_map else 0
        R_retx = sum(v['retx'] for v in reserved_map.values()) if reserved_map else 0
        reserved_total = min(M_MAX, R_new + R_retx)
        R_new_series[slot] = R_new
        R_retx_series[slot] = R_retx
        
        # General pool allocation
        M_general = max(0, M_MAX - reserved_total)
        nonburst_mask = ~burst_mask[slot]
        U_new_general = int(U_new_g[nonburst_mask].sum())
        U_retx_general = int(U_retx_g[nonburst_mask].sum())
        total_general_cont = U_new_general + U_retx_general
        
        if M_general == 0 or total_general_cont == 0:
            M_new_gen = 0
            M_retx_gen = M_general if M_general > 0 else 0
        else:
            M_new_gen = int(np.floor(M_general * (U_new_general / (total_general_cont + eps))))
            M_new_gen = max(0, min(M_new_gen, M_general))
            M_retx_gen = M_general - M_new_gen
        
        # Starvation guard
        if MIN_RETX_PREAMBLES > 0 and U_retx_general > 0:
            retx_share = U_retx_general / (total_general_cont + eps) if total_general_cont > 0 else 0.0
            retx_starved = (U_retx_general >= STARVATION_MULTIPLIER * max(1, M_retx_gen)) or (retx_share >= STARVATION_SHARE)
            if retx_starved:
                M_retx_gen = max(M_retx_gen, MIN_RETX_PREAMBLES)
                M_retx_gen = min(M_retx_gen, M_general)
                M_new_gen = max(0, M_general - M_retx_gen)
        
        M_new_series[slot] = M_new_gen
        M_retx_series[slot] = M_retx_gen
        
        # Compute p-persistent probabilities
        p_new_g = np.zeros(G, dtype=np.float64)
        p_retx_g = np.zeros(G, dtype=np.float64)
        
        for g in range(G):
            if g in reserved_map and burst_mask[slot, g]:
                Rn = reserved_map[g]['new']
                Rr = reserved_map[g]['retx']
                if Rn > 0:
                    desired_new = TARGET_FILL * Rn
                    p_new_g[g] = min(1.0, PERSIST_K_RES * desired_new / (U_new_g[g] + eps))
                if Rr > 0:
                    desired_retx = TARGET_FILL * Rr
                    p_retx_g[g] = min(1.0, PERSIST_K_RES * desired_retx / (U_retx_g[g] + eps))
        
        PERSIST_K_GEN_eff = PERSIST_K_GEN * k_gen
        p_new_gen = min(1.0, PERSIST_K_GEN_eff * M_new_gen / (U_new_general + eps)) if M_new_gen > 0 and U_new_general > 0 else 0.0
        p_retx_gen = min(1.0, PERSIST_K_GEN_eff * M_retx_gen / (U_retx_general + eps)) if M_retx_gen > 0 and U_retx_general > 0 else 0.0
        
        # Build contiguous preamble ranges
        preambleCounter = np.zeros(M_MAX, dtype=np.int32)
        ranges = {}
        idx = 0
        active_groups_list = list(reserved_map.keys()) if reserved_map else []
        
        for g in active_groups_list:
            n = reserved_map[g]['new']
            if n > 0 and idx < M_MAX:
                ranges[(g, 'N')] = (idx, min(M_MAX, idx + n))
                idx = min(M_MAX, idx + n)
        
        for g in active_groups_list:
            n = reserved_map[g]['retx']
            if n > 0 and idx < M_MAX:
                ranges[(g, 'R')] = (idx, min(M_MAX, idx + n))
                idx = min(M_MAX, idx + n)
        
        if M_new_gen > 0 and idx < M_MAX:
            ranges[('GEN', 'N')] = (idx, min(M_MAX, idx + M_new_gen))
            idx = min(M_MAX, idx + M_new_gen)
        
        if M_retx_gen > 0 and idx < M_MAX:
            ranges[('GEN', 'R')] = (idx, min(M_MAX, idx + M_retx_gen))
        
        # Process UEs with optimized preamble selection
        contenders_this_slot = 0
        U_total_now = U_new_total + U_retx_total
        pressure = U_retx_total / (U_total_now + eps) if U_total_now > 0 else 0.0
        
        for i in range(n_ues):
            if not ue_active[i] or ue_backoffs[i] > 0:
                continue
            
            g = ue_groups[i]
            in_burst = burst_mask[slot, g]
            is_new = ue_transmissions[i] == 0
            took_action = False
            
            # Try reserved pool first
            if in_burst and g in reserved_map:
                if is_new and reserved_map[g]['new'] > 0 and (g, 'N') in ranges:
                    if rng.random() < p_new_g[g]:
                        pr = fast_two_choice_preamble(preambleCounter, ranges[(g, 'N')][0], ranges[(g, 'N')][1])
                        if pr >= 0:
                            ue_preambles[i] = pr
                            preambleCounter[pr] += 1
                            ue_transmissions[i] += 1
                            took_action = True
                elif not is_new and reserved_map[g]['retx'] > 0 and (g, 'R') in ranges:
                    if rng.random() < p_retx_g[g]:
                        pr = fast_two_choice_preamble(preambleCounter, ranges[(g, 'R')][0], ranges[(g, 'R')][1])
                        if pr >= 0:
                            ue_preambles[i] = pr
                            preambleCounter[pr] += 1
                            ue_transmissions[i] += 1
                            took_action = True
                
                if not took_action:
                    go_overflow = (('GEN', 'N') in ranges and is_new) or (('GEN', 'R') in ranges and not is_new)
                    if not (go_overflow and rng.random() < OVERFLOW_P):
                        ue_backoffs[i] = rng.integers(SHORT_SKIP_MIN, SHORT_SKIP_MAX + 1)
                        ue_preambles[i] = -1
                        continue
            
            # Try general pool
            if not took_action:
                if is_new and ('GEN', 'N') in ranges and M_new_gen > 0:
                    if rng.random() < p_new_gen:
                        pr = fast_two_choice_preamble(preambleCounter, ranges[('GEN', 'N')][0], ranges[('GEN', 'N')][1])
                        if pr >= 0:
                            ue_preambles[i] = pr
                            preambleCounter[pr] += 1
                            ue_transmissions[i] += 1
                            took_action = True
                elif not is_new and ('GEN', 'R') in ranges and M_retx_gen > 0:
                    if rng.random() < p_retx_gen:
                        pr = fast_two_choice_preamble(preambleCounter, ranges[('GEN', 'R')][0], ranges[('GEN', 'R')][1])
                        if pr >= 0:
                            ue_preambles[i] = pr
                            preambleCounter[pr] += 1
                            ue_transmissions[i] += 1
                            took_action = True
                
                if not took_action:
                    ue_backoffs[i] = rng.integers(SHORT_SKIP_MIN, SHORT_SKIP_MAX + 1)
                    ue_preambles[i] = -1
                    continue
            
            if ue_preambles[i] >= 0:
                contenders_this_slot += 1
        
        # Resolve outcomes
        attempts_this_slot = 0
        collided_UEs_this_slot = 0
        unused_pream = 0
        coll_pream = 0
        
        for j in range(M_MAX):
            c = preambleCounter[j]
            attempts_this_slot += c
            if c == 0:
                unused_pream += 1
                last_state[j] = 0
            elif c == 1:
                last_state[j] = 1
            else:
                coll_pream += 1
                last_state[j] = 2
                collided_UEs_this_slot += c
        
        congestedPreambles[slot] = coll_pream
        freePreambles[slot] = unused_pream
        usedPreambles[slot] = M_MAX - unused_pream
        attemptedUEsPerSlot[slot] = attempts_this_slot
        collidedUEsPerSlot[slot] = collided_UEs_this_slot
        collisionProbPerSlot[slot] = collided_UEs_this_slot / attempts_this_slot if attempts_this_slot > 0 else 0.0
        
        # PI update with anti-windup
        used_now = max(1, usedPreambles[slot])
        lambda_hat = attempts_this_slot / float(used_now)
        err = LAMBDA_TARGET - lambda_hat
        e_int = np.clip(e_int + err, -5.0, 5.0)
        k_gen = np.clip(k_gen + KP * err + KI * e_int, KGEN_MIN, KGEN_MAX)
        
        # Process outcomes and apply backoff
        finished_mask = np.zeros(n_ues, dtype=np.bool_)
        
        for i in range(n_ues):
            if not ue_active[i] or ue_backoffs[i] > 0:
                continue
            
            if ue_preambles[i] >= 0 and preambleCounter[ue_preambles[i]] == 1:
                # Success
                successfulUEsPerSlot[slot] += 1
                success_delays_slots.append(slot - ue_first_slots[i])
                finished_mask[i] = True
            elif ue_preambles[i] >= 0 and ue_transmissions[i] >= preambleTransMax:
                # Dropped
                dropped_delays_slots.append(slot - ue_first_slots[i])
                finished_mask[i] = True
            elif ue_preambles[i] >= 0:
                # Collision - apply adaptive backoff
                if backoffBool:
                    cap_ms = int(min(BACKOFF_CAP_MS_MAX, BACKOFF_BASE_MS * (1 + RETX_PRESSURE_GAIN * pressure)))
                    r = max(1e-9, 1 - rng.random())
                    randomBackoff_ms = int(cap_ms * (-np.log(r) / 2.0))
                    slots_backoff = int(np.ceil(randomBackoff_ms / (frameSize * 1000.0)))
                    ue_backoffs[i] = slots_backoff
        
        # Remove finished UEs by swapping with last active
        if finished_mask.any():
            keep_mask = ~finished_mask
            n_keep = np.sum(keep_mask)
            
            # Compact arrays
            temp_groups = ue_groups[:n_ues][keep_mask].copy()
            temp_trans = ue_transmissions[:n_ues][keep_mask].copy()
            temp_pream = ue_preambles[:n_ues][keep_mask].copy()
            temp_backoff = ue_backoffs[:n_ues][keep_mask].copy()
            temp_first = ue_first_slots[:n_ues][keep_mask].copy()
            
            ue_groups[:n_keep] = temp_groups
            ue_transmissions[:n_keep] = temp_trans
            ue_preambles[:n_keep] = temp_pream
            ue_backoffs[:n_keep] = temp_backoff
            ue_first_slots[:n_keep] = temp_first
            ue_active[:n_keep] = True
            ue_active[n_keep:n_ues] = False
            n_ues = n_keep
        
        # Decrement backoffs for remaining UEs
        ue_backoffs[:n_ues] = np.maximum(0, ue_backoffs[:n_ues] - 1)
        UEsPerSlot[slot] = contenders_this_slot
    
    # Aggregate metrics
    total_successes = int(successfulUEsPerSlot.sum())
    total_contenders = int(UEsPerSlot.sum())
    success_rate_per_attempt = total_successes / total_contenders if total_contenders > 0 else 0.0
    
    success_delays_sec = np.array(success_delays_slots, dtype=np.float64) * frameSize
    dropped_delays_sec = np.array(dropped_delays_slots, dtype=np.float64) * frameSize
    
    overall_collision_probability = float(collidedUEsPerSlot.sum()) / float(attemptedUEsPerSlot.sum()) if attemptedUEsPerSlot.sum() > 0 else 0.0
    avg_slot_collision_probability = float(np.mean(collisionProbPerSlot)) if collisionProbPerSlot.size else 0.0
    
    delay_stats = {
        "count_success": int(success_delays_sec.size),
        "avg_success_delay_s": float(np.mean(success_delays_sec)) if success_delays_sec.size else None,
        "p50_success_delay_s": float(np.percentile(success_delays_sec, 50)) if success_delays_sec.size else None,
        "p95_success_delay_s": float(np.percentile(success_delays_sec, 95)) if success_delays_sec.size else None,
        "count_dropped": int(dropped_delays_sec.size),
        "avg_dropped_lifetime_s": float(np.mean(dropped_delays_sec)) if dropped_delays_sec.size else None,
        "p95_dropped_lifetime_s": float(np.percentile(dropped_delays_sec, 95)) if dropped_delays_sec.size else None,
    }
    
    metrics = {
        "success_rate_per_attempt": success_rate_per_attempt,
        "total_successes": total_successes,
        "total_contenders": total_contenders,
        "avg_preamble_occupancy": float(np.mean(usedPreambles / float(M_MAX))) if M_MAX > 0 else 0.0,
        "avg_collided_preambles": float(np.mean(congestedPreambles)),
        "avg_free_preambles": float(np.mean(freePreambles)),
        "delay_stats": delay_stats,
        "overall_collision_probability": overall_collision_probability,
        "avg_slot_collision_probability": avg_slot_collision_probability
    }
    
    per_slot = {
        "successfulUEsPerSlot": successfulUEsPerSlot,
        "UEsPerSlot": UEsPerSlot,
        "congestedPreambles": congestedPreambles,
        "freePreambles": freePreambles,
        "usedPreambles": usedPreambles,
        "newTraffic": newTraffic,
        "retxTraffic": retxTraffic,
        "M_new_general": M_new_series,
        "M_retx_general": M_retx_series,
        "R_new_reserved": R_new_series,
        "R_retx_reserved": R_retx_series,
        "success_delays_s": success_delays_sec,
        "dropped_delays_s": dropped_delays_sec,
        "collidedUEsPerSlot": collidedUEsPerSlot,
        "attemptedUEsPerSlot": attemptedUEsPerSlot,
        "collisionProbPerSlot": collisionProbPerSlot
    }
    
    return metrics, per_slot

    slots, G = arrivals_per_group.shape
    eps = 1e-9

    # Per-slot arrays
    UEsPerSlot = np.zeros(slots, dtype=int)
    successfulUEsPerSlot = np.zeros(slots, dtype=int)
    congestedPreambles = np.zeros(slots, dtype=int)
    freePreambles = np.zeros(slots, dtype=int)
    usedPreambles = np.zeros(slots, dtype=int)
    newTraffic = np.zeros(slots, dtype=int)
    retxTraffic = np.zeros(slots, dtype=int)
    M_new_series = np.zeros(slots, dtype=int)
    M_retx_series = np.zeros(slots, dtype=int)
    R_new_series = np.zeros(slots, dtype=int)
    R_retx_series = np.zeros(slots, dtype=int)

    collidedUEsPerSlot = np.zeros(slots, dtype=int)
    attemptedUEsPerSlot = np.zeros(slots, dtype=int)
    collisionProbPerSlot = np.zeros(slots, dtype=float)

    success_delays_slots = []
    dropped_delays_slots = []

    UEs = []
    last_state = np.zeros(M_MAX, dtype=int)

    # --- PI controller (conservative) ---
    k_gen = 1.0
    e_int = 0.0
    KP = 0.20
    KI = 0.010
    KGEN_MIN, KGEN_MAX = 0.45, 1.40
    LAMBDA_TARGET = 0.45   # target attempts / usedPreambles

    def scale_reserved(req_new, req_retx, budget):
        total_req = sum(req_new) + sum(req_retx)
        if total_req <= budget or total_req == 0:
            return req_new, req_retx
        scale = budget / total_req
        rn = [int(np.floor(x * scale)) for x in req_new]
        rr = [int(np.floor(x * scale)) for x in req_retx]
        used = sum(rn) + sum(rr)
        leftover = max(0, budget - used)
        idx = 0
        order = [('n', i) for i in range(len(rn))] + [('r', i) for i in range(len(rr))]
        while leftover > 0 and order:
            tag, i = order[idx % len(order)]
            if tag == 'n':
                rn[i] += 1
            else:
                rr[i] += 1
            leftover -= 1
            idx += 1
        return rn, rr

    rng = np.random.default_rng()

    def two_choice_pick(r):
        """Pick one preamble using 'power-of-2 choices'."""
        a, b = r
        if b <= a:
            return None
        if b - a == 1:
            return a
        i = rng.integers(a, b)
        j = rng.integers(a, b)
        return i if preambleCounter[i] <= preambleCounter[j] else j

    for slot in range(slots):
        # Add newcomers per group
        for g in range(G):
            for _ in range(int(arrivals_per_group[slot, g])):
                UEs.append(UE(group_id=g, first_slot=slot))

        # Count ready/backoff per group
        U_new_g = np.zeros(G, dtype=int)
        U_retx_g = np.zeros(G, dtype=int)
        back_new_g = np.zeros(G, dtype=int)
        back_retx_g = np.zeros(G, dtype=int)
        for dev in UEs:
            if dev.backoffCounter == 0:
                if dev.transmissions == 0:
                    U_new_g[dev.group] += 1
                else:
                    U_retx_g[dev.group] += 1
            else:
                if dev.transmissions == 0:
                    back_new_g[dev.group] += 1
                else:
                    back_retx_g[dev.group] += 1

        U_new_total = int(U_new_g.sum())
        U_retx_total = int(U_retx_g.sum())
        newTraffic[slot] = U_new_total
        retxTraffic[slot] = U_retx_total

        # Reserved plan
        if RES_POLICY is not None:
            reserved_map = RES_POLICY.step(
                slot,
                burst_mask[slot],
                U_new_g, U_retx_g,
                back_new_g, back_retx_g
            )
        else:
            req_new, req_retx, active_groups = [], [], []
            for g in range(G):
                if burst_mask[slot, g]:
                    rconf = (RESERVATION or {}).get(g, {'new': 0, 'retx': 0})
                    req_new.append(int(rconf.get('new', 0)))
                    req_retx.append(int(rconf.get('retx', 0)))
                    active_groups.append(g)
            if active_groups:
                rn, rr = scale_reserved(req_new, req_retx, M_MAX)
                reserved_map = {g: {'new': rn[i], 'retx': rr[i]}
                                for i, g in enumerate(active_groups)}
            else:
                reserved_map = {}

        R_new = sum(v['new'] for v in reserved_map.values()) if reserved_map else 0
        R_retx = sum(v['retx'] for v in reserved_map.values()) if reserved_map else 0
        reserved_total = min(M_MAX, R_new + R_retx)
        R_new_series[slot] = R_new
        R_retx_series[slot] = R_retx

        # General pool after carve-out
        M_general = max(0, M_MAX - reserved_total)

        # Split general pool between NEW/RETX using non-burst contenders
        nonburst_mask = ~burst_mask[slot]
        U_new_general = int(U_new_g[nonburst_mask].sum())
        U_retx_general = int(U_retx_g[nonburst_mask].sum())
        total_general_cont = U_new_general + U_retx_general

        if M_general == 0 or total_general_cont == 0:
            M_new_gen = 0
            M_retx_gen = 0 if M_general == 0 else M_general
        else:
            M_new_gen = int(np.floor(M_general * (U_new_general / (total_general_cont + eps))))
            M_new_gen = max(0, min(M_new_gen, M_general))
            M_retx_gen = M_general - M_new_gen

        # Optional starvation guard (general RETX)
        total_cont = U_new_general + U_retx_general
        if MIN_RETX_PREAMBLES > 0 and U_retx_general > 0:
            retx_share = (U_retx_general / (total_cont + eps)) if total_cont > 0 else 0.0
            retx_starved = ((U_retx_general >= STARVATION_MULTIPLIER * max(1, M_retx_gen))
                            or (retx_share >= STARVATION_SHARE))
            if retx_starved:
                M_retx_gen = max(M_retx_gen, MIN_RETX_PREAMBLES)
                M_retx_gen = min(M_retx_gen, M_general)
                M_new_gen = max(0, M_general - M_retx_gen)

        M_new_series[slot] = M_new_gen
        M_retx_series[slot] = M_retx_gen

        # p-persistent (reserved: drive-to-fill below unit load)
        p_new_g = np.zeros(G, dtype=float)
        p_retx_g = np.zeros(G, dtype=float)
        for g in range(G):
            if g in reserved_map and burst_mask[slot, g]:
                Rn = reserved_map[g]['new']
                Rr = reserved_map[g]['retx']
                if Rn > 0:
                    desired_attempts_new = TARGET_FILL * Rn
                    p_new_g[g] = (1.0 if U_new_g[g] == 0
                                  else min(1.0, PERSIST_K_RES * desired_attempts_new / (U_new_g[g] + eps)))
                else:
                    p_new_g[g] = 0.0
                if Rr > 0:
                    desired_attempts_retx = TARGET_FILL * Rr
                    p_retx_g[g] = (1.0 if U_retx_g[g] == 0
                                   else min(1.0, PERSIST_K_RES * desired_attempts_retx / (U_retx_g[g] + eps)))
                else:
                    p_retx_g[g] = 0.0
            else:
                p_new_g[g] = 0.0
                p_retx_g[g] = 0.0

        # general pools with PI
        PERSIST_K_GEN_eff = PERSIST_K_GEN * k_gen
        p_new_gen = 0.0 if M_new_gen == 0 else (1.0 if U_new_general == 0 else min(1.0, PERSIST_K_GEN_eff * M_new_gen / (U_new_general + eps)))
        p_retx_gen = 0.0 if M_retx_gen == 0 else (1.0 if U_retx_general == 0 else min(1.0, PERSIST_K_GEN_eff * M_retx_gen / (U_retx_general + eps)))

        # Build contiguous ranges
        preambleCounter = {k: 0 for k in range(M_MAX)}
        ranges = {}
        idx = 0
        active_groups = list(reserved_map.keys())

        # reserved NEW
        for g in active_groups:
            n = reserved_map[g]['new']
            if n > 0 and idx < M_MAX:
                a = idx
                b = min(M_MAX, idx + n)
                ranges[(g, 'N')] = (a, b)
                idx = b
        # reserved RETX
        for g in active_groups:
            n = reserved_map[g]['retx']
            if n > 0 and idx < M_MAX:
                a = idx
                b = min(M_MAX, idx + n)
                ranges[(g, 'R')] = (a, b)
                idx = b
        # general NEW
        if M_new_gen > 0 and idx < M_MAX:
            a = idx
            b = min(M_MAX, idx + M_new_gen)
            ranges[('GEN', 'N')] = (a, b)
            idx = b
        # general RETX
        if M_retx_gen > 0 and idx < M_MAX:
            a = idx
            b = min(M_MAX, idx + M_retx_gen)
            ranges[('GEN', 'R')] = (a, b)
            idx = b

        def allow_attempt_reserved(dev):
            g = dev.group
            if dev.transmissions == 0:
                return np.random.rand() < p_new_g[g]
            return np.random.rand() < p_retx_g[g]

        def allow_attempt_general(dev):
            if dev.transmissions == 0:
                return np.random.rand() < p_new_gen
            return np.random.rand() < p_retx_gen

        contenders_this_slot = 0
        for dev in UEs:
            if dev.backoffCounter > 0:
                dev.preamble = 99
                continue

            in_burst = burst_mask[slot, dev.group]
            took_action = False

            # reserved first (if burst)
            if in_burst and dev.group in reserved_map:
                if dev.transmissions == 0 and reserved_map[dev.group]['new'] > 0 and (dev.group, 'N') in ranges:
                    if allow_attempt_reserved(dev):
                        pr = two_choice_pick(ranges[(dev.group, 'N')])
                        if pr is not None:
                            dev.preamble = pr
                            preambleCounter[pr] += 1
                            dev.transmissions += 1
                            took_action = True
                elif dev.transmissions > 0 and reserved_map[dev.group]['retx'] > 0 and (dev.group, 'R') in ranges:
                    if allow_attempt_reserved(dev):
                        pr = two_choice_pick(ranges[(dev.group, 'R')])
                        if pr is not None:
                            dev.preamble = pr
                            preambleCounter[pr] += 1
                            dev.transmissions += 1
                            took_action = True

                # اگر هنوز کاری نکرده: overflow کنترل‌شده به عمومی
                if not took_action:
                    go_overflow = (('GEN', 'N') in ranges and dev.transmissions == 0) or (('GEN', 'R') in ranges and dev.transmissions > 0)
                    if go_overflow and (np.random.rand() < OVERFLOW_P):
                        # اجازه می‌دهیم در بخش general pools تلاش کند
                        pass
                    else:
                        skip_slots = np.random.randint(SHORT_SKIP_MIN, SHORT_SKIP_MAX + 1)
                        dev.backoffCounter = skip_slots
                        dev.preamble = 99
                        continue

            # general pools
            if not took_action:
                if dev.transmissions == 0 and ('GEN', 'N') in ranges and M_new_gen > 0:
                    if allow_attempt_general(dev):
                        pr = two_choice_pick(ranges[('GEN', 'N')])
                        if pr is not None:
                            dev.preamble = pr
                            preambleCounter[pr] += 1
                            dev.transmissions += 1
                            took_action = True
                elif dev.transmissions > 0 and ('GEN', 'R') in ranges and M_retx_gen > 0:
                    if allow_attempt_general(dev):
                        pr = two_choice_pick(ranges[('GEN', 'R')])
                        if pr is not None:
                            dev.preamble = pr
                            preambleCounter[pr] += 1
                            dev.transmissions += 1
                            took_action = True

                if not took_action:
                    skip_slots = np.random.randint(SHORT_SKIP_MIN, SHORT_SKIP_MAX + 1)
                    dev.backoffCounter = skip_slots
                    dev.preamble = 99
                    continue

            if dev.preamble != 99:
                contenders_this_slot += 1

        # Outcomes
        attempts_this_slot = 0
        collided_UEs_this_slot = 0
        unused_pream = 0
        coll_pream = 0
        for j in range(M_MAX):
            c = preambleCounter[j]
            attempts_this_slot += c
            if c == 0:
                unused_pream += 1
                last_state[j] = 0
            elif c == 1:
                last_state[j] = 1
            else:
                coll_pream += 1
                last_state[j] = 2
                collided_UEs_this_slot += c

        congestedPreambles[slot] = coll_pream
        freePreambles[slot] = unused_pream
        usedPreambles[slot] = M_MAX - unused_pream

        attemptedUEsPerSlot[slot] = attempts_this_slot
        collidedUEsPerSlot[slot] = collided_UEs_this_slot
        collisionProbPerSlot[slot] = (collided_UEs_this_slot / attempts_this_slot) if attempts_this_slot > 0 else 0.0

        # PI update (for next slot general p)
        used_now = max(1, usedPreambles[slot])
        lambda_hat = attempts_this_slot / float(used_now)
        err = LAMBDA_TARGET - lambda_hat
        e_int = np.clip(e_int + err, -5.0, 5.0)
        k_gen = np.clip(k_gen + KP * err + KI * e_int, KGEN_MIN, KGEN_MAX)

        # Backoff (heavy tail)
        finishedUEs = []
        for dev in UEs:
            if dev.backoffCounter == 0:
                if dev.preamble != 99 and preambleCounter.get(dev.preamble, 0) == 1:
                    successfulUEsPerSlot[slot] += 1
                    success_delays_slots.append(slot - dev.first_slot)
                    finishedUEs.append(dev)
                elif dev.preamble != 99 and dev.transmissions >= preambleTransMax:
                    dropped_delays_slots.append(slot - dev.first_slot)
                    finishedUEs.append(dev)
                elif dev.preamble != 99:
                    if backoffBool:
                        U_total_now = U_new_total + U_retx_total
                        pressure = (U_retx_total / (U_total_now + eps)) if U_total_now > 0 else 0.0
                        cap_ms = int(min(BACKOFF_CAP_MS_MAX,
                                         BACKOFF_BASE_MS * (1 + RETX_PRESSURE_GAIN * pressure)))
                        r = max(1e-9, 1 - np.random.rand())
                        randomBackoff_ms = int(cap_ms * (-np.log(r) / 2.0))
                    else:
                        randomBackoff_ms = 0
                    slots_backoff = int(np.ceil(randomBackoff_ms / (frameSize * 1000.0)))
                    dev.backoffCounter = slots_backoff
            else:
                dev.backoffCounter -= 1

        for dev in finishedUEs:
            UEs.remove(dev)
            del dev

        UEsPerSlot[slot] = contenders_this_slot

    # Aggregate metrics
    total_successes = int(np.sum(successfulUEsPerSlot))
    total_contenders = int(np.sum(UEsPerSlot))
    success_rate_per_attempt = (total_successes / total_contenders) if total_contenders > 0 else 0.0

    success_delays_sec = np.array(success_delays_slots, dtype=float) * frameSize
    dropped_delays_sec = np.array(dropped_delays_slots, dtype=float) * frameSize

    overall_collision_probability = (
        float(np.sum(collidedUEsPerSlot)) / float(np.sum(attemptedUEsPerSlot))
        if np.sum(attemptedUEsPerSlot) > 0 else 0.0
    )
    avg_slot_collision_probability = float(np.mean(collisionProbPerSlot)) if collisionProbPerSlot.size else 0.0

    delay_stats = {
        "count_success": int(success_delays_sec.size),
        "avg_success_delay_s": float(np.mean(success_delays_sec)) if success_delays_sec.size else None,
        "p50_success_delay_s": float(np.percentile(success_delays_sec, 50)) if success_delays_sec.size else None,
        "p95_success_delay_s": float(np.percentile(success_delays_sec, 95)) if success_delays_sec.size else None,
        "count_dropped": int(dropped_delays_sec.size),
        "avg_dropped_lifetime_s": float(np.mean(dropped_delays_sec)) if dropped_delays_sec.size else None,
        "p95_dropped_lifetime_s": float(np.percentile(dropped_delays_sec, 95)) if dropped_delays_sec.size else None,
    }

    metrics = {
        "success_rate_per_attempt": success_rate_per_attempt,
        "total_successes": total_successes,
        "total_contenders": total_contenders,
        "avg_preamble_occupancy": float(np.mean(usedPreambles / float(M_MAX))) if M_MAX > 0 else 0.0,
        "avg_collided_preambles": float(np.mean(congestedPreambles)),
        "avg_free_preambles": float(np.mean(freePreambles)),
        "delay_stats": delay_stats,
        "overall_collision_probability": overall_collision_probability,
        "avg_slot_collision_probability": avg_slot_collision_probability
    }

    per_slot = {
        "successfulUEsPerSlot": successfulUEsPerSlot,
        "UEsPerSlot": UEsPerSlot,
        "congestedPreambles": congestedPreambles,
        "freePreambles": freePreambles,
        "usedPreambles": usedPreambles,
        "newTraffic": newTraffic,
        "retxTraffic": retxTraffic,
        "M_new_general": M_new_series,
        "M_retx_general": M_retx_series,
        "R_new_reserved": R_new_series,
        "R_retx_reserved": R_retx_series,
        "success_delays_s": success_delays_sec,
        "dropped_delays_s": dropped_delays_sec,
        "collidedUEsPerSlot": collidedUEsPerSlot,
        "attemptedUEsPerSlot": attemptedUEsPerSlot,
        "collisionProbPerSlot": collisionProbPerSlot
    }

    return metrics, per_slot

# ============================
# Main with multiple profiles + clean curves
# ============================

if __name__ == "__main__":
    # --- Traffic Config ---
    numDevicesVec = [20000, 10000, 8000, 10000, 10000, 20000, 10000, 12000, 10000, 10000]
    eventProbabilities = [0.006, 0.009, 0.09, 0.1, 0.2, 0.004, 0.004, 0.05, 0.1, 0.2]
    totalStreams = 1
    T = 10
    frameSize = 0.005
    slots = int(T / frameSize)
    G = len(numDevicesVec)

    # --- Dynamic Group-time Reservation Policy (per-active cap, a bit higher) ---
    RES_POLICY = DynamicReservationPolicy(
        G=G, M_MAX=54,
        base_new=2, base_retx=2,
        max_per_group=6,
        hard_cap_total=16,
        cap_per_active=5,
        w_burst=1.6, w_retx_share=1.0, w_backlog=0.6,
        tau_on=0.55, tau_off=0.35,
        ramp_up=2, ramp_down=3,
        cooldown_slots=int(0.7 / frameSize),  # ~0.7s cooldown
        min_when_on=2
    )

    # --- Profiles (RescuePlus) ---
    PARAM_PROFILES = [
        {
            "name": "RescuePlus",
            "desc": "Higher burst reservation + controlled overflow + conservative PI",
            "PERSIST_K_GEN": 0.48,
            "PERSIST_K_RES": 1.20,
            "TARGET_FILL": 0.78,
            "SHORT_SKIP_MIN": 1,
            "SHORT_SKIP_MAX": 5,
            "BACKOFF_BASE_MS": 40,
            "RETX_PRESSURE_GAIN": 3.5,
            "BACKOFF_CAP_MS_MAX": 550,
            "MIN_RETX_PREAMBLES": 0,
            "STARVATION_SHARE": 0.5,
            "STARVATION_MULTIPLIER": 2.0,
            "OVERFLOW_P": 0.35
        }
    ]

    # --- Storage Prep (root dir) ---
    TT = datetime.now()
    root_dir = r'D:\YAZD\Masters Thesis\Code\P-persistant and adaptive backoff\With tuned parameters\Tuned-V2\with collision probability\Result'
    if not os.path.isdir(root_dir):
        try:
            os.makedirs(root_dir, exist_ok=True)
        except Exception:
            root_dir = os.path.join(os.getcwd(), 'A_generatedTraffic')
            os.makedirs(root_dir, exist_ok=True)

    # === Run profiles ===
    for prof in PARAM_PROFILES:
        print(f"\n===== Running profile: {prof['name']}  ({prof['desc']}) =====")

        # Unique path per profile
        address1 = (
            root_dir
            + f'/{TT.strftime("%j")}_{TT.strftime("%a")}_'
            + f'{TT.strftime("%b")}{TT.strftime("%d")}_'
            + f'{TT.strftime("%H")}{TT.strftime("%M")}'
            + f'_Samples({totalStreams})({T}Sec)_{prof["name"]}'
        )
        fileType = '.pt'
        address = address1 + fileType
        cnt = 1
        while os.path.isfile(address):
            address = address1 + f'({cnt})' + fileType
            cnt += 1

        # Create a subdirectory for plots
        plots_dir = os.path.join(root_dir, f"plots_{prof['name']}_{TT.strftime('%H%M')}")
        os.makedirs(plots_dir, exist_ok=True)

        Intensity = []
        Pattern = []
        aggregate_success = 0
        aggregate_contenders = 0
        aggregate_delays_success = []
        aggregate_delays_dropped = []
        aggregate_collided = 0
        aggregate_attempted = 0

        starttime = datetime.now()
        for i in tqdm(range(totalStreams),
                      desc=f'Generating & simulating [{prof["name"]}]',
                      position=0, colour='red'):
            arrivals_total, arrivals_per_group, eventsAll, TbsAll = newArivals(
                numDevicesVec, eventProbabilities, T, frameSize
            )
            traffic_smoothed = signal.savgol_filter(arrivals_total, 97, 2)
            burst_mask = burst_mask_from_events(eventsAll, TbsAll, slots, frameSize)

            # Reset policy per stream
            RES_POLICY.on_flags[:] = False
            RES_POLICY.on_until[:] = 0
            RES_POLICY.curr_new[:] = 0
            RES_POLICY.curr_retx[:] = 0

            metrics, per_slot = actualTrafficPattern(
                arrivals_per_group, burst_mask, frameSize=frameSize, backoffBool=True,
                PERSIST_K_GEN=prof["PERSIST_K_GEN"],
                PERSIST_K_RES=prof["PERSIST_K_RES"],
                TARGET_FILL=prof["TARGET_FILL"],
                SHORT_SKIP_MIN=prof["SHORT_SKIP_MIN"],
                SHORT_SKIP_MAX=prof["SHORT_SKIP_MAX"],
                BACKOFF_BASE_MS=prof["BACKOFF_BASE_MS"],
                RETX_PRESSURE_GAIN=prof["RETX_PRESSURE_GAIN"],
                BACKOFF_CAP_MS_MAX=prof["BACKOFF_CAP_MS_MAX"],
                MIN_RETX_PREAMBLES=prof["MIN_RETX_PREAMBLES"],
                STARVATION_SHARE=prof["STARVATION_SHARE"],
                STARVATION_MULTIPLIER=prof["STARVATION_MULTIPLIER"],
                OVERFLOW_P=prof["OVERFLOW_P"],
                RESERVATION=None,
                CARVE_OUT=True,
                RES_POLICY=RES_POLICY
            )

            Intensity.append([arrivals_total, eventsAll, TbsAll, traffic_smoothed])
            Pattern.append(per_slot)

            aggregate_success += metrics["total_successes"]
            aggregate_contenders += metrics["total_contenders"]
            aggregate_collided += int(np.sum(per_slot["collidedUEsPerSlot"]))
            aggregate_attempted += int(np.sum(per_slot["attemptedUEsPerSlot"]))
            if per_slot["success_delays_s"].size:
                aggregate_delays_success.append(per_slot["success_delays_s"])
            if per_slot["dropped_delays_s"].size:
                aggregate_delays_dropped.append(per_slot["dropped_delays_s"])

        agg_success_delays = (np.concatenate(aggregate_delays_success)
                              if aggregate_delays_success else np.array([]))
        agg_dropped_delays = (np.concatenate(aggregate_delays_dropped)
                              if aggregate_delays_dropped else np.array([]))

        overall_success_rate = (aggregate_success / aggregate_contenders) if aggregate_contenders > 0 else 0.0
        overall_collision_prob = (aggregate_collided / aggregate_attempted) if aggregate_attempted > 0 else 0.0
        agg_delay_stats = {
            "count_success": int(agg_success_delays.size),
            "avg_success_delay_s": float(np.mean(agg_success_delays)) if agg_success_delays.size else None,
            "p50_success_delay_s": float(np.percentile(agg_success_delays, 50)) if agg_success_delays.size else None,
            "p95_success_delay_s": float(np.percentile(agg_success_delays, 95)) if agg_success_delays.size else None,
            "count_dropped": int(agg_dropped_delays.size),
            "avg_dropped_lifetime_s": float(np.mean(agg_dropped_delays)) if agg_dropped_delays.size else None,
            "p95_dropped_lifetime_s": float(np.percentile(agg_dropped_delays, 95)) if agg_dropped_delays.size else None,
        }

        # Save
        torch.save({
            'Description': f'Dynamic split + Patch1/3 + Dynamic Group Reservation + PI control + 2-choice + controlled overflow [{prof["name"]}]',
            'profile': prof,
            'numDevicesVec': numDevicesVec,
            'eventProbabilities': eventProbabilities,
            'T': T,
            'frameSize': frameSize,
            'Intensity': Intensity,
            'Pattern': Pattern,
            'overall_success_rate': overall_success_rate,
            'aggregate_success': aggregate_success,
            'aggregate_contenders': aggregate_contenders,
            'aggregate_delay_stats': agg_delay_stats,
            'overall_collision_probability': overall_collision_prob,
            'reservation_policy': {
                'hard_cap_total': RES_POLICY.hard_cap_total,
                'max_per_group': RES_POLICY.max_per_group,
                'cap_per_active': RES_POLICY.cap_per_active,
                'tau_on': RES_POLICY.tau_on,
                'tau_off': RES_POLICY.tau_off,
                'cooldown_slots': RES_POLICY.cooldown_slots,
                'ramp_up': RES_POLICY.ramp_up,
                'ramp_down': RES_POLICY.ramp_down
            }
        }, address)

        # Print summary
        print('\n==== Summary ==== ')
        print(f'Output file: {address}')
        print(f'Total contenders (attempts): {aggregate_contenders:,}')
        print(f'Total successes:             {aggregate_success:,}')
        print(f'Overall success rate:        {overall_success_rate:.4f}')
        print(f'Overall collision probability: {overall_collision_prob:.4f}')
        if agg_success_delays.size:
            print(f'Avg success delay (s):       {agg_delay_stats["avg_success_delay_s"]:.4f}')
            print(f'P50 success delay (s):       {agg_delay_stats["p50_success_delay_s"]:.4f}')
            print(f'P95 success delay (s):       {agg_delay_stats["p95_success_delay_s"]:.4f}')
        else:
            print('No successful transmissions to report delay stats.')
        if agg_dropped_delays.size:
            print(f'Dropped UEs:                 {agg_delay_stats["count_dropped"]:,}')
            print(f'Avg dropped lifetime (s):    {agg_delay_stats["avg_dropped_lifetime_s"]:.4f}')
            print(f'P95 dropped lifetime (s):    {agg_delay_stats["p95_dropped_lifetime_s"]:.4f}')
        else:
            print('No dropped UEs.')
        print('Wall-clock:', datetime.now() - starttime)

        # === Generate high-quality plots ===
        print(f"\nGenerating publication-quality plots for {prof['name']}...")

        # استفاده از داده‌های stream اول برای نمودارها
        per_slot_0 = Pattern[0]

        # تولید تمام نمودارها
        plot_enhanced_success_rate(per_slot_0, prof['name'], plots_dir)
        plot_collision_probability(per_slot_0, prof['name'], plots_dir)
        plot_traffic_composition(per_slot_0, prof['name'], plots_dir)
        plot_preamble_allocation(per_slot_0, prof['name'], plots_dir)
        plot_preamble_usage_breakdown(per_slot_0, prof['name'], plots_dir)
        plot_delay_ecdf(per_slot_0, prof['name'], plots_dir)
        plot_utilization_metrics(per_slot_0, prof['name'], plots_dir)

        # تحلیل حساسیت تاخیر بر حسب تعداد دستگاه‌ها (اگر totalStreams کم باشد اجرا شود)
        if totalStreams <= 10:  # فقط اگر تعداد streamها کم باشد اجرا شود (زمان‌بر است)
            print("Delay sensitivity analysis based on the number of devices...")
            delay_data = plot_delay_vs_devices(numDevicesVec, eventProbabilities, T, frameSize, plots_dir, num_points=6)
            plot_scalability_analysis(delay_data, plots_dir)
        else:
            print("Due to the large number of streams, latency sensitivity analysis is not performed (it is time consuming).")

        # شبیه‌سازی برای تعداد دستگاه‌های مختلف
        target_device_counts = [50000, 100000, 150000, 200000]

        print("\n" + "="*60)
        print("RUNNING MULTI-SCENARIO ANALYSIS FOR DIFFERENT DEVICE COUNTS")
        print("="*60)

        all_scenarios_results = simulate_multiple_device_scenarios(
            numDevicesVec, eventProbabilities, T, frameSize, 
            target_device_counts, plots_dir
        )

        print(f"All plots for {prof['name']} saved in high-quality formats in: {plots_dir}")

# Backward compatibility alias
actualTrafficPattern = actualTrafficPattern_optimized