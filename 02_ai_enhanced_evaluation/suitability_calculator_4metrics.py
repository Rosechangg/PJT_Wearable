"""
Suitability Calculator for Exoskeleton Tasks (4 Metrics with Hilbert Transform)
4개 척도 기반 적합도 계산기: Power Sign 제거, Hilbert 변환 적용

척도:
1. Torque Stability (Hilbert 변환 적용)
2. Angle Smoothness (Hilbert 변환 적용)
3. ROM 적합도 (가동범위)
4. ST (Human Torque 안정성)
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import re
from scipy.signal import hilbert

def load_mmt_info():
    """MMT 등급 정보 로드"""
    try:
        mmt_df = pd.read_csv('data/user_info_patient.csv')
        mmt_dict = dict(zip(mmt_df['Number'], mmt_df['MMT_grade']))
        return mmt_dict
    except:
        print("Warning: MMT info not found, using default values")
        return {}

def parse_filename(filename):
    """파일명에서 환자 번호와 Task 번호 추출"""
    # 001_Task1.csv -> (1, 1)
    match = re.match(r'(\d+)_Task(\d+)\.csv', filename)
    if match:
        patient_id = int(match.group(1))
        task_id = int(match.group(2))
        return patient_id, task_id
    return None, None

def load_patient_data():
    """모든 환자 데이터 로드"""
    patient_folder = 'data/patient_data'
    csv_files = glob.glob(os.path.join(patient_folder, '*.csv'))

    all_data = []

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        patient_id, task_id = parse_filename(filename)

        if patient_id is None or task_id is None:
            continue

        try:
            df = pd.read_csv(file_path)
            # 공백 제거
            df.columns = df.columns.str.strip()

            # 필요한 컬럼만 필터링 (Status 컬럼 포함)
            required_cols = ['Repeat', 'Weight', 'Elbow Angle', 'Current']
            if all(col in df.columns for col in required_cols):
                df['Patient'] = patient_id
                df['Task'] = task_id

                # 유효한 데이터만 필터링 (첫 번째 행 제외)
                df = df[df['Repeat'] != 'Repeat']  # 헤더 중복 제거
                df = df.dropna(subset=['Repeat'])

                if len(df) > 0:
                    # 데이터 타입 변환
                    df['Repeat'] = pd.to_numeric(df['Repeat'], errors='coerce')
                    df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
                    df['Elbow Angle'] = pd.to_numeric(df['Elbow Angle'], errors='coerce')
                    df['Current'] = pd.to_numeric(df['Current'], errors='coerce')

                    # Human Torque와 Motor Torque 계산 (임시)
                    # 실제 계산 공식에 맞게 수정 필요
                    df['Human_Torque'] = df['Weight'] * 0.1  # 임시 공식
                    df['Motor_Torque'] = df['Current'] * 0.01  # 임시 공식

                    all_data.append(df)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def build_training_stats(all_data):
    """전체 훈련 데이터에서 ROM 분위수 계산"""
    rom_values = []

    for _, group in all_data.groupby(['Patient', 'Task', 'Repeat']):
        if len(group) < 5:
            continue
        angle = group['Elbow Angle'].values.astype(float)
        rom = float(np.max(angle) - np.min(angle))
        rom_values.append(rom)

    if not rom_values:
        return {'ROM_p10': 0, 'ROM_p90': 1}

    return {
        'ROM_p10': np.percentile(rom_values, 10),
        'ROM_p90': np.percentile(rom_values, 90)
    }

def calculate_torque_stability_hilbert(group):
    """
    1. Torque Stability with Hilbert Transform
    Hilbert 변환을 통한 인스턴트 amplitude로 안정성 측정
    """
    human_torque = group['Human_Torque'].values.astype(float)

    if len(human_torque) < 5:
        return 0.5

    # Hilbert 변환으로 analytic signal 계산
    analytic_signal = hilbert(human_torque)
    instantaneous_amplitude = np.abs(analytic_signal)

    # 인스턴트 amplitude의 변동성 계산
    amplitude_std = np.std(instantaneous_amplitude)
    amplitude_mean = np.mean(instantaneous_amplitude)

    if amplitude_mean < 1e-6:
        return 0.5

    # Coefficient of Variation 기반 안정성
    cv = amplitude_std / (amplitude_mean + 1e-6)

    # 안정성으로 변환 (낮은 CV = 높은 안정성)
    stability = 1.0 / (1.0 + cv)

    return float(stability)

def calculate_angle_smoothness(group):
    """2. Angle Smoothness (각도 스무스함) - Jerk 기반"""
    angle = group['Elbow Angle'].values.astype(float)

    # 각속도 (1차 미분)
    angular_velocity = np.diff(angle, prepend=angle[0])

    # 각가속도 (2차 미분)
    angular_acceleration = np.diff(angular_velocity, prepend=angular_velocity[0])

    # Jerk (3차 미분)
    jerk = np.diff(angular_acceleration, prepend=angular_acceleration[0])

    # Jerk의 분산 (낮을수록 부드러움)
    jerk_variance = np.var(jerk)

    # 스무스함으로 변환
    smoothness = 1.0 / (1.0 + jerk_variance)

    return float(smoothness)

def calculate_rom_suitability(group, stats):
    """3. ROM 적합도 (가동범위)"""
    angle = group['Elbow Angle'].values.astype(float)
    rom = float(np.max(angle) - np.min(angle))

    # 분위수 기반 정규화
    rom_p10 = stats['ROM_p10']
    rom_p90 = stats['ROM_p90']

    if rom_p90 - rom_p10 > 0:
        normalized_rom = (rom - rom_p10) / (rom_p90 - rom_p10)
        normalized_rom = np.clip(normalized_rom, 0.0, 1.0)
    else:
        normalized_rom = 0.5

    return float(normalized_rom)

def calculate_st_suitability(group):
    """4. ST (Human Torque 안정성)"""
    human_torque = group['Human_Torque'].values.astype(float)

    # Median과 MAD 계산
    median_torque = np.median(human_torque)
    mad = np.median(np.abs(human_torque - median_torque))

    # 스케일 정규화
    scale = max(np.abs(median_torque), 1e-6)

    # ST 계산
    st = 1.0 / (1.0 + (mad / scale))

    return float(st)

def calculate_motion_suitability(group, stats):
    """단일 모션(Repeat)에 대한 4개 척도 계산"""
    if len(group) < 5:
        return None

    # 4개 척도 계산
    torque_stability = calculate_torque_stability_hilbert(group)
    angle_smoothness = calculate_angle_smoothness(group)
    rom_suitability = calculate_rom_suitability(group, stats)
    st_suitability = calculate_st_suitability(group)

    return {
        'torque_stability': torque_stability,
        'angle_smoothness': angle_smoothness,
        'rom_suitability': rom_suitability,
        'st_suitability': st_suitability
    }

def calculate_adaptive_weights(patient_results):
    """
    환자별 적응형 가중치 계산
    - Task간 분산이 큰 척도에 높은 가중치
    - 상관관계가 높은 척도들에 페널티
    """
    metrics = ['torque_stability', 'angle_smoothness', 'rom_suitability', 'st_suitability']

    # 각 척도별 Task간 값들 추출
    metric_values = {}
    for metric in metrics:
        values = [patient_results[task][metric] for task in [1, 2, 3]]
        metric_values[metric] = values

    # 1. 분산 기반 가중치 (변별력)
    variances = {}
    for metric in metrics:
        variances[metric] = np.var(metric_values[metric])

    # 분산을 0-1로 정규화하여 가중치로 사용
    max_var = max(variances.values()) if max(variances.values()) > 0 else 1e-6
    variance_weights = {metric: variances[metric] / max_var for metric in metrics}

    # 2. 상관관계 기반 페널티
    correlation_matrix = np.corrcoef([metric_values[metric] for metric in metrics])
    correlation_penalties = {}

    for i, metric in enumerate(metrics):
        # 자기 자신 제외하고 다른 척도들과의 평균 상관계수
        other_correlations = [abs(correlation_matrix[i][j]) for j in range(len(metrics)) if i != j]
        avg_correlation = np.mean(other_correlations)

        # 상관관계가 높을수록 페널티 (1 - 상관계수)
        correlation_penalties[metric] = 1 - avg_correlation

    # 3. 최종 가중치 = 분산 가중치 × 상관 페널티
    final_weights = {}
    for metric in metrics:
        final_weights[metric] = variance_weights[metric] * correlation_penalties[metric]

    # 가중치 정규화 (합이 1이 되도록)
    total_weight = sum(final_weights.values())
    if total_weight > 0:
        final_weights = {metric: weight / total_weight for metric, weight in final_weights.items()}
    else:
        # 모든 가중치가 0이면 균등 분배
        final_weights = {metric: 0.25 for metric in metrics}

    return final_weights

def calculate_patient_suitability(all_data, patient_id, stats):
    """환자별 전체 적합도 계산 (적응형 가중화 적용)"""

    # 각 Task별 결과 계산
    patient_results = {}
    for task in [1, 2, 3]:
        task_result = calculate_task_suitability_base(all_data, patient_id, task, stats)
        if task_result is not None:
            patient_results[task] = task_result

    if len(patient_results) < 3:
        return None

    # 적응형 가중치 계산
    weights = calculate_adaptive_weights(patient_results)

    metrics = ['torque_stability', 'angle_smoothness', 'rom_suitability', 'st_suitability']

    # 환자별 z-score 표준화
    normalized_results = {}
    for task in [1, 2, 3]:
        normalized_results[task] = {}

        # 각 척도별로 3개 Task 값들의 z-score 계산
        for metric in metrics:
            task_values = [patient_results[t][metric] for t in [1, 2, 3]]
            mean_val = np.mean(task_values)
            std_val = np.std(task_values)

            if std_val > 1e-6:
                z_score = (patient_results[task][metric] - mean_val) / std_val
            else:
                z_score = 0

            normalized_results[task][metric] = z_score

    # 가중 합계 계산
    weighted_scores = {}
    for task in [1, 2, 3]:
        weighted_score = sum(normalized_results[task][metric] * weights[metric] for metric in metrics)
        weighted_scores[task] = weighted_score

    # 환자별 0-1 정규화
    min_score = min(weighted_scores.values())
    max_score = max(weighted_scores.values())

    final_results = {}
    for task in [1, 2, 3]:
        if max_score - min_score > 1e-6:
            normalized_score = (weighted_scores[task] - min_score) / (max_score - min_score)
        else:
            normalized_score = 0.5  # 모든 값이 같으면 중간값

        # 원래 척도 값들과 함께 저장
        final_results[task] = {
            **patient_results[task],
            'overall_suitability': normalized_score,
            'weighted_score': weighted_scores[task],
            'weights': weights  # 디버깅용
        }

    return final_results

def calculate_task_suitability_base(all_data, patient_id, task, stats):
    """기본 Task 적합도 계산 (가중화 전)"""
    task_data = all_data[(all_data['Patient'] == patient_id) & (all_data['Task'] == task)]

    if len(task_data) == 0:
        return None

    motion_results = []

    # Repeat별(모션별) 계산
    for repeat in task_data['Repeat'].unique():
        if pd.isna(repeat):
            continue
        motion_data = task_data[task_data['Repeat'] == repeat]
        motion_result = calculate_motion_suitability(motion_data, stats)

        if motion_result is not None:
            motion_results.append(motion_result)

    if not motion_results:
        return None

    # Task별 평균 계산
    task_avg = {}
    for metric in ['torque_stability', 'angle_smoothness', 'rom_suitability', 'st_suitability']:
        values = [result[metric] for result in motion_results]
        task_avg[metric] = np.mean(values)

    return task_avg

def main():
    """메인 함수: 모든 환자의 적합도 계산"""
    print("=== Exoskeleton Task Suitability Calculator (4 Metrics with Hilbert) ===")

    # MMT 정보 로드
    mmt_dict = load_mmt_info()

    # 모든 환자 데이터 로드
    print("Loading patient data...")
    all_data = load_patient_data()

    if len(all_data) == 0:
        print("Error: No valid patient data found")
        return

    print(f"Loaded data for {all_data['Patient'].nunique()} patients")

    # ROM 통계 계산
    stats = build_training_stats(all_data)
    print(f"ROM statistics: P10={stats['ROM_p10']:.2f}, P90={stats['ROM_p90']:.2f}")

    # 각 환자별 적합도 계산 (적응형 가중화 적용)
    results = []
    patients = sorted(all_data['Patient'].unique())

    for patient_id in patients:
        mmt_grade = mmt_dict.get(patient_id, 4)  # 기본값 4
        print(f"\n=== Patient {patient_id:03d} (MMT: {mmt_grade}) ===")

        # 환자별 적응형 가중화 적합도 계산
        patient_result = calculate_patient_suitability(all_data, patient_id, stats)

        if patient_result is not None:
            # 가중치 출력 (디버깅)
            weights = patient_result[1]['weights']
            print(f"  Adaptive Weights:")
            print(f"    Torque Stability: {weights['torque_stability']:.3f}")
            print(f"    Angle Smoothness: {weights['angle_smoothness']:.3f}")
            print(f"    ROM Suitability: {weights['rom_suitability']:.3f}")
            print(f"    ST Suitability: {weights['st_suitability']:.3f}")

            # Task별 결과 출력
            for task in [1, 2, 3]:
                task_result = patient_result[task]
                print(f"  Task {task}:")
                print(f"    Torque Stability: {task_result['torque_stability']:.4f}")
                print(f"    Angle Smoothness: {task_result['angle_smoothness']:.4f}")
                print(f"    ROM Suitability: {task_result['rom_suitability']:.4f}")
                print(f"    ST Suitability: {task_result['st_suitability']:.4f}")
                print(f"    Weighted Score: {task_result['weighted_score']:.4f}")
                print(f"    Overall Suitability: {task_result['overall_suitability']:.4f}")

                # 결과 저장 (weights 제외)
                result_row = {
                    'Patient': patient_id,
                    'MMT': mmt_grade,
                    'Task': task,
                    'torque_stability': task_result['torque_stability'],
                    'angle_smoothness': task_result['angle_smoothness'],
                    'rom_suitability': task_result['rom_suitability'],
                    'st_suitability': task_result['st_suitability'],
                    'weighted_score': task_result['weighted_score'],
                    'overall_suitability': task_result['overall_suitability']
                }
                results.append(result_row)
        else:
            print(f"  No valid data for patient {patient_id}")

    # 결과를 CSV로 저장
    if results:
        results_df = pd.DataFrame(results)

        # Task 이름 추가
        task_names = {1: 'Stiff', 2: 'Compliant', 3: 'Resistant'}
        results_df['Task_Name'] = results_df['Task'].map(task_names)

        # 값 반올림
        results_df['torque_stability'] = results_df['torque_stability'].round(3)
        results_df['angle_smoothness'] = results_df['angle_smoothness'].round(3)
        results_df['rom_suitability'] = results_df['rom_suitability'].round(3)
        results_df['st_suitability'] = results_df['st_suitability'].round(3)
        results_df['weighted_score'] = results_df['weighted_score'].round(3)
        results_df['overall_suitability'] = results_df['overall_suitability'].round(3)

        # 컬럼 순서 재정렬
        results_df = results_df[['Patient', 'MMT', 'Task', 'Task_Name', 'torque_stability',
                               'angle_smoothness', 'rom_suitability', 'st_suitability',
                               'weighted_score', 'overall_suitability']]

        # 컬럼명 변경
        results_df.columns = ['Patient', 'MMT', 'Task', 'Task_Name', 'Torque_Stability',
                            'Angle_Smoothness', 'ROM_Suitability', 'ST_Suitability',
                            'Weighted_Score', 'Overall_Suitability']

        # CSV 저장
        results_df.to_csv('results/suitability_results_4metrics_adaptive.csv', index=False)
        print(f"\n=== Results saved to suitability_results_4metrics_adaptive.csv ===")

        # 요약 통계
        print(f"\n=== Summary Statistics ===")
        for task in [1, 2, 3]:
            task_data = results_df[results_df['Task'] == task]
            if len(task_data) > 0:
                avg_suitability = task_data['Overall_Suitability'].mean()
                print(f"Task {task} Average Suitability: {avg_suitability:.4f}")

if __name__ == "__main__":
    main()