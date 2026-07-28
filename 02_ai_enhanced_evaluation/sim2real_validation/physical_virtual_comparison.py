"""
Physical vs Virtual Data Comparison Tool
5가지 메트릭을 계산하여 R-square 및 p-value를 산출
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import hilbert
from datetime import datetime


# ============ Configuration ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHYSICAL_DIR = os.path.join(BASE_DIR, "data", "physical")
VIRTUAL_DIR = os.path.join(BASE_DIR, "data", "virtual")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")


# ============ Metric Calculation Functions ============

def calculate_jerk(angles, time_interval=0.02):
    """
    Angle Smoothness 계산: 각속도의 표준편차
    값이 작을수록 부드러운 움직임 (흔들림이 적음)
    """
    if len(angles) < 2:
        return np.nan

    # 각속도 계산 (1차 미분)
    angular_velocity = np.diff(angles) / time_interval

    # 각속도의 표준편차 (흔들림 정도)
    smoothness = np.std(angular_velocity)
    return smoothness


def calculate_rom(angles):
    """
    ROM 계산: P90 - P10 (Robust Range of Motion)
    이상치에 강건한 범위 측정
    """
    if len(angles) < 2:
        return np.nan

    p10 = np.percentile(angles, 10)
    p90 = np.percentile(angles, 90)
    rom = p90 - p10
    return rom


def calculate_rms(weights):
    """
    RMS 계산: Root Mean Square
    """
    if len(weights) < 1:
        return np.nan

    rms = np.sqrt(np.mean(np.array(weights) ** 2))
    return rms


def calculate_hilbert_envelope(weights):
    """
    Hilbert 변환: 신호의 envelope(포락선) 평균
    흔들림 정도를 나타냄
    """
    if len(weights) < 2:
        return np.nan

    analytic_signal = hilbert(weights)
    envelope = np.abs(analytic_signal)
    return np.mean(envelope)


def calculate_sign_sum(weights):
    """
    Sign Sum: 값의 합에 sign 함수 적용
    양수/음수 우세 방향 확인
    """
    if len(weights) < 1:
        return np.nan

    total_sum = np.sum(weights)
    return np.sign(total_sum)


# ============ Data Loading Functions ============

def load_physical_data(task_num, max_samples=None, target_count=36):
    """
    Physical 데이터 로드
    각 사람별 Repeat 1,2,3의 Flexion 데이터만 추출
    target_count에 맞춰서 데이터 수 조정
    Returns: list of dicts
    """
    task_dir = os.path.join(PHYSICAL_DIR, f"task{task_num}")
    csv_files = sorted(glob.glob(os.path.join(task_dir, "*.csv")))

    all_data = []

    for csv_file in csv_files:
        person_id = os.path.basename(csv_file).split('_')[0]
        df = pd.read_csv(csv_file)

        # Flexion만 필터링
        df_flexion = df[df['U_Status'] == 'Flexion']

        # Repeat 1, 2, 3 각각 추출
        for repeat_num in [1, 2, 3]:
            df_repeat = df_flexion[df_flexion['Repeat'] == repeat_num]

            if len(df_repeat) > 0:
                angles = df_repeat['Elbow Angle'].values
                weights = df_repeat['Weight'].values
                # 시간 인덱스 생성 (0.02초 간격 가정)
                timestamps = np.arange(len(angles)) * 0.02

                all_data.append({
                    'person_id': person_id,
                    'repeat': repeat_num,
                    'angles': angles,
                    'weights': weights,
                    'timestamps': timestamps,
                    'source': os.path.basename(csv_file)
                })

                if max_samples and len(all_data) >= max_samples:
                    return all_data

    # target_count에 맞춰서 조정 (부족하면 반복, 초과하면 자르기)
    if target_count and len(all_data) < target_count:
        original_len = len(all_data)
        while len(all_data) < target_count:
            idx = len(all_data) % original_len
            all_data.append(all_data[idx].copy())

    if target_count:
        all_data = all_data[:target_count]

    return all_data


def load_virtual_data(task_num):
    """
    Virtual 데이터 로드
    각 파일별로 데이터 추출 (radian 그대로 사용)
    Returns: list of dicts with angles, weights, timestamps
    """
    task_dir = os.path.join(VIRTUAL_DIR, f"task{task_num}")
    csv_files = sorted(glob.glob(os.path.join(task_dir, "*.csv")))

    all_data = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        # radian 그대로 사용 (변환 없음)
        angles_rad = df['elbow_angle_rad'].values

        weights = df['inter_force'].values
        timestamps = df['timestamp_sec'].values if 'timestamp_sec' in df.columns else np.arange(len(angles_rad)) * 0.02

        all_data.append({
            'file_name': os.path.basename(csv_file),
            'angles': angles_rad,  # radian 그대로
            'weights': weights,
            'timestamps': timestamps
        })

    return all_data


# ============ Metric Calculation ============

def calculate_all_metrics(angles, weights):
    """
    5가지 메트릭 모두 계산
    """
    return {
        'jerk': calculate_jerk(angles),
        'rom': calculate_rom(angles),
        'rms': calculate_rms(weights),
        'hilbert_envelope': calculate_hilbert_envelope(weights),
        'sign_sum': calculate_sign_sum(weights)
    }


# ============ Visualization ============

def plot_comparison(virtual_data, physical_data, task_num, output_dir, sample_idx=0):
    """
    선택된 데이터 쌍의 4개 그래프 출력
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f'Task {task_num}: Virtual vs Physical Data Comparison (Sample {sample_idx + 1})', fontsize=14)

    # Virtual Angle
    ax1 = axes[0, 0]
    ax1.plot(virtual_data['timestamps'], virtual_data['angles'], 'b-', linewidth=1.5)
    ax1.set_title('Virtual - Elbow Angle')
    ax1.set_xlabel('Time (sec)')
    ax1.set_ylabel('Angle (degree)')
    ax1.grid(True, alpha=0.3)

    # Virtual Weight (inter_force)
    ax2 = axes[0, 1]
    ax2.plot(virtual_data['timestamps'], virtual_data['weights'], 'b-', linewidth=1.5)
    ax2.set_title('Virtual - Inter Force (Weight)')
    ax2.set_xlabel('Time (sec)')
    ax2.set_ylabel('Force')
    ax2.grid(True, alpha=0.3)

    # Physical Angle
    ax3 = axes[1, 0]
    ax3.plot(physical_data['timestamps'], physical_data['angles'], 'r-', linewidth=1.5)
    ax3.set_title(f"Physical - Elbow Angle ({physical_data['person_id']} Repeat {physical_data['repeat']})")
    ax3.set_xlabel('Time (sec)')
    ax3.set_ylabel('Angle (degree)')
    ax3.grid(True, alpha=0.3)

    # Physical Weight
    ax4 = axes[1, 1]
    ax4.plot(physical_data['timestamps'], physical_data['weights'], 'r-', linewidth=1.5)
    ax4.set_title(f"Physical - Weight ({physical_data['person_id']} Repeat {physical_data['repeat']})")
    ax4.set_xlabel('Time (sec)')
    ax4.set_ylabel('Weight')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 저장
    plot_path = os.path.join(output_dir, f'comparison_plot_task{task_num}_sample{sample_idx + 1}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nGraph saved: {plot_path}")

    plt.show()


# ============ Statistical Analysis ============

def calculate_r_square(physical_metrics, virtual_metrics):
    """
    각 메트릭별 R 및 R-square 계산
    """
    results = {}
    metric_names = ['jerk', 'rom', 'rms', 'hilbert_envelope', 'sign_sum']

    for metric in metric_names:
        phys_values = [m[metric] for m in physical_metrics]
        virt_values = [m[metric] for m in virtual_metrics]

        # NaN 제거
        valid_pairs = [(p, v) for p, v in zip(phys_values, virt_values)
                       if not np.isnan(p) and not np.isnan(v)]

        if len(valid_pairs) >= 2:
            phys_arr = np.array([p[0] for p in valid_pairs])
            virt_arr = np.array([p[1] for p in valid_pairs])

            # x값(Physical)이 모두 동일한지 체크 (y값은 동일해도 계산 가능)
            if np.all(phys_arr == phys_arr[0]):
                # 값이 모두 동일하면 회귀 계산 불가
                results[metric] = {
                    'r': np.nan,
                    'r_squared': np.nan,
                    'n_samples': len(valid_pairs),
                    'slope': np.nan,
                    'intercept': np.nan,
                    'note': 'All values identical - regression not possible'
                }
            else:
                # 선형 회귀
                slope, intercept, r_value, _, _ = stats.linregress(phys_arr, virt_arr)
                r_squared = r_value ** 2

                results[metric] = {
                    'r': r_value,
                    'r_squared': r_squared,
                    'n_samples': len(valid_pairs),
                    'slope': slope,
                    'intercept': intercept
                }
        else:
            results[metric] = {
                'r': np.nan,
                'r_squared': np.nan,
                'n_samples': len(valid_pairs),
                'slope': np.nan,
                'intercept': np.nan
            }

    return results


# ============ Main Function ============

def main():
    # Task 선택
    print("=" * 50)
    print("Physical vs Virtual Data Comparison Tool")
    print("=" * 50)
    print("\nSelect Task Number:")
    print("  1 - Task 1")
    print("  3 - Task 3")

    while True:
        task_input = input("\nEnter task number (1 or 3): ").strip()
        if task_input in ['1', '3']:
            task_num = int(task_input)
            break
        print("Invalid input. Please enter 1 or 3.")

    print(f"\n>>> Processing Task {task_num}...")

    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Virtual 데이터 로드
    print("\nLoading Virtual data...")
    virtual_data = load_virtual_data(task_num)
    num_virtual = len(virtual_data)
    print(f"  Found {num_virtual} virtual file(s)")

    if num_virtual == 0:
        print("No virtual data found. Exiting.")
        return

    # Physical 데이터 로드 (36개 고정)
    print("\nLoading Physical data...")
    physical_data = load_physical_data(task_num, max_samples=None, target_count=36)
    num_physical = len(physical_data)
    print(f"  Loaded {num_physical} physical sample(s)")

    if num_physical == 0:
        print("No physical data found. Exiting.")
        return

    # 실제 비교할 개수
    num_compare = min(num_virtual, num_physical)

    if num_virtual < num_physical:
        print(f"\n[WARNING] Virtual files ({num_virtual}) < Physical samples ({num_physical})")
        print(f"          Run augment_virtual_data.py to generate more virtual files!")

    print(f"\n>>> Comparing {num_compare} sample pairs...")

    # 그래프 출력할 샘플 선택
    print(f"\n>>> Select sample for graph (1-{num_compare}):")
    while True:
        sample_input = input(f"Enter sample number (1-{num_compare}, default=1): ").strip()
        if sample_input == '':
            sample_idx = 0
            break
        try:
            sample_idx = int(sample_input) - 1
            if 0 <= sample_idx < num_compare:
                break
            print(f"Invalid input. Please enter a number between 1 and {num_compare}.")
        except ValueError:
            print(f"Invalid input. Please enter a number between 1 and {num_compare}.")

    print(f"\n>>> Plotting sample {sample_idx + 1} comparison...")
    print(f"    Physical: {physical_data[sample_idx]['source']} Repeat {physical_data[sample_idx]['repeat']}")
    print(f"    Virtual:  {virtual_data[sample_idx]['file_name']}")
    plot_comparison(virtual_data[sample_idx], physical_data[sample_idx], task_num, OUTPUT_DIR, sample_idx)

    # 메트릭 계산
    print("\n>>> Calculating metrics for all samples...")

    physical_metrics = []
    virtual_metrics = []
    detail_rows = []

    for i in range(num_compare):
        phys = physical_data[i]
        virt = virtual_data[i]

        phys_metrics = calculate_all_metrics(phys['angles'], phys['weights'])
        virt_metrics = calculate_all_metrics(virt['angles'], virt['weights'])

        physical_metrics.append(phys_metrics)
        virtual_metrics.append(virt_metrics)

        # 상세 결과 저장용
        detail_rows.append({
            'sample_index': i + 1,
            'physical_source': phys['source'],
            'physical_repeat': phys['repeat'],
            'virtual_source': virt['file_name'],
            'phys_jerk': phys_metrics['jerk'],
            'phys_rom': phys_metrics['rom'],
            'phys_rms': phys_metrics['rms'],
            'phys_hilbert': phys_metrics['hilbert_envelope'],
            'phys_sign': phys_metrics['sign_sum'],
            'virt_jerk': virt_metrics['jerk'],
            'virt_rom': virt_metrics['rom'],
            'virt_rms': virt_metrics['rms'],
            'virt_hilbert': virt_metrics['hilbert_envelope'],
            'virt_sign': virt_metrics['sign_sum'],
        })

    # 상세 메트릭 CSV 저장
    detail_df = pd.DataFrame(detail_rows)
    detail_path = os.path.join(OUTPUT_DIR, f'metrics_detail_task{task_num}.csv')
    detail_df.to_csv(detail_path, index=False)
    print(f"\nMetrics detail saved: {detail_path}")

    # R 및 R-square 계산
    print("\n>>> Calculating R and R-square...")
    results = calculate_r_square(physical_metrics, virtual_metrics)

    # 결과 출력
    print("\n" + "=" * 70)
    print(f"RESULTS - Task {task_num} (n={num_compare} samples)")
    print("=" * 70)

    metric_display_names = {
        'jerk': 'Jerk (Angle Smoothness)',
        'rom': 'ROM (Range of Motion)',
        'rms': 'RMS (Weight)',
        'hilbert_envelope': 'Hilbert Envelope (Oscillation)',
        'sign_sum': 'Sign Sum (Direction)'
    }

    result_rows = []
    for metric, display_name in metric_display_names.items():
        res = results[metric]
        print(f"\n{display_name}:")
        print(f"  r:         {res['r']:.6f}" if not np.isnan(res['r']) else "  r:         N/A")
        print(f"  R-squared: {res['r_squared']:.6f}" if not np.isnan(res['r_squared']) else "  R-squared: N/A")
        print(f"  Samples:   {res['n_samples']}")

        result_rows.append({
            'metric': metric,
            'display_name': display_name,
            'r': res['r'],
            'r_squared': res['r_squared'],
            'n_samples': res['n_samples'],
            'slope': res['slope'],
            'intercept': res['intercept']
        })

    # 결과 CSV 저장
    result_df = pd.DataFrame(result_rows)
    result_path = os.path.join(OUTPUT_DIR, f'comparison_results_task{task_num}.csv')
    result_df.to_csv(result_path, index=False)
    print(f"\n{'=' * 70}")
    print(f"Results saved: {result_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
