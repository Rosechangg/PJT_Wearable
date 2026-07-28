"""
Ordinal Learning for Resistance-Assistance Continuum:
Task mapping: Resistance (-1) -> Compliant (0) -> Very Compliant (+0.5) -> Stiff (+1)

Normal: Task1=Very Compliant(+0.5), Task2=Compliant(0), Task3=Stiff(+1)
Patient: Task1=Stiff(+1), Task2=Compliant(0), Task3=Resistance(-1)

Features: Base features + Human Torque change + Angle change
Target: Single ordinal value on Resistance-Assistance continuum (-1 to +1)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score, roc_auc_score
from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import LeaveOneOut
import math
from scipy.stats import pearsonr                                             
import warnings
warnings.filterwarnings('ignore')

# GPU 설정 (RTX 5090 호환성)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

device = torch.device("cuda:0")
print(f"Using device: {device}")
print(f"CUDA capability: {torch.cuda.get_device_capability()}")

# 추가 메트릭 함수들
def concordance_index(y_true, y_pred):
    """C-index (Harrell's concordance index) 계산"""
    n = len(y_true)
    concordant = 0
    total = 0

    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                total += 1
                if (y_true[i] - y_true[j]) * (y_pred[i] - y_pred[j]) > 0:
                    concordant += 1
                elif (y_true[i] - y_true[j]) * (y_pred[i] - y_pred[j]) == 0:
                    concordant += 0.5

    return concordant / total if total > 0 else 0.5

def cumulative_auc(y_true, y_pred, thresholds=None):
    """Cumulative AUC 계산"""
    if thresholds is None:
        # Resistance-Assistance continuum의 주요 임계값들
        thresholds = [-0.5, 0.0, 0.5]  # Resistance, Compliant, Stiff 경계

    aucs = []
    for threshold in thresholds:
        y_binary = (y_true >= threshold).astype(int)
        if len(np.unique(y_binary)) == 2:  # 두 클래스가 모두 존재할 때만
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(y_binary, y_pred)
                aucs.append(auc)
            except:
                aucs.append(0.5)
        else:
            aucs.append(0.5)

    return np.mean(aucs) if aucs else 0.5

# 데이터 로드
def load_data():
    """데이터 로드 및 전처리"""
    # 정상인 데이터 (특정 번호만 선택: 1,2,3,7,10,17,21,22,23,24)
    df_normal = pd.read_csv('data/result_normal.csv')
    selected_normal_numbers = [1, 2, 3, 7, 10, 17, 21, 22, 23, 24]
    df_normal = df_normal[df_normal['Number'].isin(selected_normal_numbers)]
    df_normal['Domain'] = 'Normal'

    # 환자 데이터 (전체 10명)
    df_patient = pd.read_csv('data/result_patient.csv')
    df_patient['Domain'] = 'Patient'

    # Repeat 필터링 (1-20)
    df_normal = df_normal[(df_normal['Repeat'] >= 1) & (df_normal['Repeat'] <= 20)]
    df_patient = df_patient[(df_patient['Repeat'] >= 1) & (df_patient['Repeat'] <= 20)]

    return df_normal, df_patient

def load_mmt_info():
    """MMT 등급 정보 로드"""
    try:
        mmt_df = pd.read_csv('data/user_info_patient.csv')
        mmt_dict = dict(zip(mmt_df['Number'], mmt_df['MMT_grade']))
        return mmt_dict
    except:
        print("Warning: MMT info not found, using default values")
        return {}

def robust_norm(x, lo, hi):
    """Robust normalization using percentiles"""
    return float(np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0))

def compute_segment_metrics(group):
    """세그먼트의 기본 메트릭 계산"""
    angle = np.round(group['Angle'].values.astype(float)).astype(int)  # 정수값으로 반올림
    ang_vel = np.diff(angle, prepend=angle[0])
    ang_acc = np.diff(ang_vel, prepend=ang_vel[0])  # 각가속도
    motor = group['Motor_Torque'].values.astype(float)
    human = group['Human_Torque'].values.astype(float)

    ROM = float(np.max(angle) - np.min(angle))
    jerk = float(np.std(np.diff(ang_vel, prepend=ang_vel[0])))

    # 각도 스무스함 Sθ: jerk 기반 (낮을수록 부드러움)
    angle_jerk = np.diff(ang_acc, prepend=ang_acc[0])  # 각 jerk (3차 미분)
    angle_smoothness = 1.0 / (1.0 + np.std(angle_jerk))  # 역변환으로 높을수록 부드러움

    HTR = float(np.mean(np.abs(human) / (np.abs(human) + np.abs(motor) + 1e-8)))

    # Motor-Human Correlation
    if len(group) < 3 or np.all(motor == motor[0]) or np.all(human == human[0]):
        corr = 0.0
    else:
        c = np.corrcoef(motor, human)[0,1]
        corr = 0.0 if np.isnan(c) else float(c)

    # 저항성 지표
    prod = motor * ang_vel
    rho = float(np.mean(prod < 0))
    local_j = np.abs(np.diff(ang_vel, prepend=ang_vel[0]))
    thr = np.quantile(local_j, 0.80)
    stall = float(np.mean(local_j > thr))

    return {'ROM': ROM, 'jerk': jerk, 'angle_smoothness': angle_smoothness, 'HTR': HTR, 'corr': corr, 'rho': rho, 'stall': stall}

def build_robust_stats(train_df):
    """훈련 데이터에서 robust statistics 계산 - 변경없음"""
    vals = []
    for _, g in train_df.groupby(['Number','Task','Repeat','State']):
        if len(g) < 5: continue
        m = compute_segment_metrics(g)
        vals.append(m)

    if not vals:
        return {'ROM_p10':0,'ROM_p90':1,'jerk_p10':0,'jerk_p90':1,'r_p10':0,'r_p90':1}

    tmp = pd.DataFrame(vals)
    r_raw = 0.7*tmp['rho'] + 0.3*tmp['stall']
    return {
        'ROM_p10': tmp['ROM'].quantile(0.10), 'ROM_p90': tmp['ROM'].quantile(0.90),
        'jerk_p10': tmp['jerk'].quantile(0.10), 'jerk_p90': tmp['jerk'].quantile(0.90),
        'r_p10': r_raw.quantile(0.10), 'r_p90': r_raw.quantile(0.90),
    }

def _calc_metrics(group):
    """기본 메트릭 + 연속 power_sign 계산"""
    base_metrics = compute_segment_metrics(group)

    # 추가 지표들
    human = group['Human_Torque'].values.astype(float)
    angle = np.round(group['Angle'].values.astype(float)).astype(int)  # 정수값으로 반올림
    ang_vel = np.diff(angle, prepend=angle[0])
    motor = group['Motor_Torque'].values.astype(float)

    # ST: MAD 기반 안정성 (scale-normalized)
    scale = max(np.median(np.abs(human)), 1e-6)  # 규모 정규화용
    mad = np.median(np.abs(human - np.median(human)))  # MAD 계산
    ST = 1.0 / (1.0 + (mad / scale))  # MAD를 규모로 정규화

    KQ = 1.0 / (1.0 + np.std(np.abs(ang_vel)))
    MHC = abs(base_metrics['corr'])
    FI = 1.0 / (1.0 + np.abs(np.polyfit(np.arange(len(human)), np.abs(human), 1)[0]))

    # Power Sign as continuous value P ∈ [-1,1] - 연속값으로 변경
    prod = motor * ang_vel
    valid = ~np.isnan(prod) & ~np.isinf(prod)
    if len(valid) < 0.3*len(prod):
        power_sign = 0.0  # 유효 샘플 부족시 중립값
    else:
        power_sign = float(np.mean(np.sign(prod[valid])))  # 연속값 평균

    # Delta Human Torque (원래 방식 복원)
    start = np.mean(human[:3]) if len(human) >= 3 else human[0]
    end = np.mean(human[-3:]) if len(human) >= 3 else human[-1]
    scale = max(np.median(np.abs(human)), 1.0)
    delta_ht = float(np.tanh((end - start)/(scale+1e-8)))

    return {**base_metrics, 'ST': ST, 'KQ': KQ, 'MHC': MHC, 'FI': FI,
            'power_sign': power_sign, 'delta_ht': delta_ht}

def generate_resistance_assistance_labels(df, stats=None, mmt_dict=None):
    """Resistance-Assistance continuum 라벨 생성"""
    rows = []

    for (num, task, rep, state), g in df.groupby(['Number', 'Task', 'Repeat', 'State']):
        if len(g) < 5:
            continue

        m = _calc_metrics(g)
        domain = g['Domain'].iloc[0]

        # Resistance-Assistance continuum 매핑
        # Normal: Task1=Very Compliant(+0.5), Task2=Compliant(0), Task3=Stiff(+1)
        # Patient: Task1=Stiff(+1), Task2=Compliant(0), Task3=Resistance(-1)

        if domain == 'Normal':
            if task == 1:  # Very Compliant
                resistance_assistance = 0.5
            elif task == 2:  # Compliant
                resistance_assistance = 0.0
            else:  # Task 3 = Stiff
                resistance_assistance = 1.0
        else:  # Patient
            if task == 1:  # Stiff
                resistance_assistance = 1.0
            elif task == 2:  # Compliant
                resistance_assistance = 0.0
            else:  # Task 3 = Resistance
                resistance_assistance = -1.0

        rows.append({
            'Number': num,
            'Task': task,
            'Repeat': rep,
            'State': state,
            'Domain': domain,
            'resistance_assistance': resistance_assistance
        })

    return pd.DataFrame(rows)

def generate_labels(df, stats=None, mmt_dict=None):
    """범용 라벨 생성 wrapper"""
    return generate_resistance_assistance_labels(df, stats, mmt_dict)

# 시퀀스 데이터 생성
def create_sequences(df, labels_df, mmt_dict=None, sequence_length=20):
    """Ordinal Learning을 위한 시퀀스 생성 - 순수 시계열만 (Task one-hot 제거)
    Features: Base(10) + Human_Torque_Change + Angle_Change + MMT = 13 features"""
    features = ['Weight', 'Angle', 'Current', 'Motion', 'Human_Torque', 'Motor_Torque',
                'Inter_Torque', 'Gravity_Torque', 'State', 'Repeat']

    sequences = []
    targets = []
    metadata = []

    for _, label_row in labels_df.iterrows():
        number = label_row['Number']
        task = label_row['Task']
        repeat = label_row['Repeat']
        state = label_row['State']

        # 해당하는 데이터 찾기 (State 포함)
        group = df[(df['Number'] == number) & (df['Task'] == task) &
                  (df['Repeat'] == repeat) & (df['State'] == state)]

        if len(group) < sequence_length:
            continue

        # MMT 등급 정보 추가
        domain = label_row['Domain']
        if domain == 'Normal':
            mmt_grade = 5.0  # 정상인은 MMT 5등급
        else:
            mmt_grade = float(mmt_dict.get(number, 4.0)) if mmt_dict else 4.0

        # 시퀀스 데이터 생성
        X_data = group[features].copy()

        # State 컬럼을 숫자로 변환 (Extension=1, Flexion=0)
        if 'State' in X_data.columns:
            X_data['State'] = X_data['State'].map({'Extension': 1, 'Flexion': 0})
            X_data['State'] = X_data['State'].fillna(0)

        # Angle을 정수로 변환
        if 'Angle' in X_data.columns:
            X_data['Angle'] = np.round(X_data['Angle'].astype(float)).astype(int)

        # 모든 컬럼이 숫자형인지 확인
        for col in X_data.columns:
            X_data[col] = pd.to_numeric(X_data[col], errors='coerce').fillna(0)

        # 추가 feature 계산: Human Torque change, Angle change
        human_torque = X_data['Human_Torque'].values
        angle = X_data['Angle'].values

        # Human Torque 변화량 (1차 차분)
        human_torque_change = np.diff(human_torque, prepend=human_torque[0])

        # Angle 변화량 (각속도)
        angle_change = np.diff(angle, prepend=angle[0])

        # 기존 features에 추가
        X_data['Human_Torque_Change'] = human_torque_change
        X_data['Angle_Change'] = angle_change

        X = X_data.values.astype(float)

        # 슬라이딩 윈도우로 여러 시퀀스 생성
        for i in range(len(X) - sequence_length + 1):
            seq = X[i:i+sequence_length]

            # MMT 등급을 각 시퀀스 스텝에 추가 (정규화된 값: 3-5 → 0-1)
            mmt_normalized = (mmt_grade - 3.0) / 2.0  # 3->0, 4->0.5, 5->1
            mmt_feature = np.full((sequence_length, 1), mmt_normalized)  # (seq_len, 1)

            # Task one-hot 제거 - 순수 시계열만으로 continuum 값 예측
            # 기존 특성 + MMT 특성만 결합
            seq_with_features = np.concatenate([seq, mmt_feature], axis=1)  # (seq_len, features+2+1)
            sequences.append(seq_with_features)

            # 타깃: Resistance-Assistance continuum 단일 ordinal 값
            targets.append(label_row['resistance_assistance'])
            metadata.append({
                'Number': number,
                'Task': task,
                'Repeat': repeat,
                'State': state,
                'Domain': label_row['Domain']
            })

    return np.array(sequences), np.array(targets), metadata

# 백본 + 다운스트림 헤드 모델
class TransformerBackbone(nn.Module):
    """Transformer 백본 (특징 추출기)"""
    def __init__(self, input_size, hidden_size=128, num_layers=4, num_heads=8, dropout=0.2):
        super(TransformerBackbone, self).__init__()

        # 입력 변환
        self.input_fc = nn.Linear(input_size, hidden_size)

        # 트랜스포머 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention Pooling Layer
        self.attention_pooling = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1)
        )

        # Temperature scaling for attention sharpening
        self.attention_temperature = nn.Parameter(torch.ones(1) * 2.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask_motor=False, mask_ratio=0.3):
        # Motor Current/Torque 마스킹 (현재 방식 유지)
        if mask_motor and self.training:
            batch_size = x.size(0)
            mask_indices = torch.rand(batch_size) < mask_ratio
            if mask_indices.any():
                x[mask_indices, :, 2] *= 0.1  # Current 약화
                x[mask_indices, :, 5] *= 0.1  # Motor_Torque 약화

        # 입력 변환
        x = self.input_fc(x)

        # 트랜스포머 인코딩
        x = self.transformer_encoder(x)

        # Attention Pooling 적용
        attention_logits = self.attention_pooling(x)
        attention_weights = torch.softmax(attention_logits / self.attention_temperature, dim=1)
        x = torch.sum(x * attention_weights, dim=1)
        x = self.dropout(x)

        return x

class OrdinalRegressor(nn.Module):
    """백본 + Resistance-Assistance continuum 순서 학습 헤드"""
    def __init__(self, input_size, hidden_size=128, num_layers=4, num_heads=8, dropout=0.2):
        super(OrdinalRegressor, self).__init__()

        # Transformer 백본
        self.backbone = TransformerBackbone(input_size, hidden_size, num_layers, num_heads, dropout)

        # Resistance-Assistance continuum 헤드 (-1 ~ +1)
        # Ordinal regression을 위한 단일 출력
        self.resistance_assistance_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Tanh()  # [-1, 1] 범위로 제한
        )

    def forward(self, x, mask_motor=False):
        # 백본 특징 추출
        features = self.backbone(x, mask_motor)

        # Resistance-Assistance continuum 예측
        resistance_assistance_out = self.resistance_assistance_head(features)

        return resistance_assistance_out

    def freeze_backbone(self):
        """백본 동결"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_top_layers(self, num_layers=2):
        """상위 레이어만 해제"""
        total_layers = len(self.backbone.transformer_encoder.layers)
        for i in range(max(0, total_layers - num_layers), total_layers):
            for param in self.backbone.transformer_encoder.layers[i].parameters():
                param.requires_grad = True

# LOSO 교차 검증
def loso_cross_validation(df_normal, df_patient, labels_df, mmt_dict):
    """Leave-One-Subject-Out 교차 검증"""
    patient_numbers = df_patient['Number'].unique()
    results = []

    for test_patient in patient_numbers:
        print(f"\n=== Testing on Patient {test_patient} ===")

        # 테스트 데이터: 선택된 환자
        test_labels = labels_df[labels_df['Number'] == test_patient]
        test_df = df_patient[df_patient['Number'] == test_patient]

        # 훈련 데이터: 나머지 환자들 + 정상인들
        train_labels = labels_df[labels_df['Number'] != test_patient]
        train_df_patient = df_patient[df_patient['Number'] != test_patient]
        train_df = pd.concat([df_normal, train_df_patient])

        if len(test_labels) == 0:
            continue

        # 훈련 데이터에서 robust stats 계산
        train_stats = build_robust_stats(train_df)

        # 라벨 생성
        train_labels = generate_labels(train_df, train_stats, mmt_dict)
        test_labels = generate_labels(test_df, train_stats, mmt_dict)

        # 시퀀스 생성
        X_train, y_train, meta_train = create_sequences(train_df, train_labels, mmt_dict)
        X_test, y_test, meta_test = create_sequences(test_df, test_labels, mmt_dict)

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        print(f"Generated {len(X_train)} train sequences, {len(X_test)} test sequences")

        # 정규화
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_train_norm = scaler.fit_transform(X_train_flat).reshape(X_train.shape)

        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        X_test_norm = scaler.transform(X_test_flat).reshape(X_test.shape)

        # 텐서 변환
        X_train_tensor = torch.FloatTensor(X_train_norm).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        X_test_tensor = torch.FloatTensor(X_test_norm).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)

        # 도메인 균형 배치 생성
        train_dataset = create_balanced_dataloader(X_train_tensor, y_train_tensor, meta_train)

        # 모델 학습 (Linear Probe + Partial Fine-tuning)
        model = OrdinalRegressor(input_size=X_train.shape[-1]).to(device)
        result = train_model_two_stage(model, train_dataset, X_test_tensor, y_test_tensor, meta_test, test_patient)
        results.append(result)

    return results

def create_balanced_dataloader(X, y, metadata, batch_size=16):
    """도메인 균형 DataLoader 생성"""
    normal_indices = [i for i, meta in enumerate(metadata) if meta['Domain'] == 'Normal']
    patient_indices = [i for i, meta in enumerate(metadata) if meta['Domain'] == 'Patient']

    # 더 작은 도메인에 맞춰 균형 조정
    min_size = min(len(normal_indices), len(patient_indices))
    if min_size == 0:
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 균형잡힌 인덱스 생성
    balanced_indices = []
    for i in range(0, min_size * 2, 2):
        if i//2 < len(normal_indices):
            balanced_indices.append(normal_indices[i//2])
        if i//2 < len(patient_indices):
            balanced_indices.append(patient_indices[i//2])

    # 배치가 너무 작으면 전체 데이터 사용
    if len(balanced_indices) < batch_size:
        balanced_indices = list(range(len(X)))

    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]

    dataset = TensorDataset(X_balanced, y_balanced)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model_two_stage(model, train_loader, X_test, y_test, meta_test, test_patient,
                         probe_epochs=30, finetune_epochs=20):
    """2단계 학습: Linear Probe + Partial Fine-tuning for Ordinal Learning"""

    huber_loss = nn.HuberLoss(delta=0.1)

    # Stage 1: Linear Probe (백본 동결)
    print("Stage 1: Linear Probe...")
    model.freeze_backbone()

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                           lr=0.001, weight_decay=1e-4)

    model.train()
    for epoch in range(probe_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            # Ordinal regression: 단일 출력
            resistance_assistance_out = model(X_batch, mask_motor=True)

            # Resistance-Assistance continuum 손실
            loss = huber_loss(resistance_assistance_out.squeeze(), y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"  Probe Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")

    # Stage 2: Partial Fine-tuning (상위 레이어 해제)
    print("Stage 2: Partial Fine-tuning...")
    model.unfreeze_top_layers(num_layers=2)

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                           lr=0.0001, weight_decay=1e-4)  # 낮은 LR

    model.train()
    for epoch in range(finetune_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            # Ordinal regression: 단일 출력
            resistance_assistance_out = model(X_batch, mask_motor=True)

            # Resistance-Assistance continuum 손실
            loss = huber_loss(resistance_assistance_out.squeeze(), y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"  Finetune Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")

    # 테스트 평가 (Ordinal Learning)
    model.eval()
    with torch.no_grad():
        resistance_assistance_pred = model(X_test)

        # 평가
        y_true = y_test.cpu().numpy()
        pred_np = resistance_assistance_pred.squeeze().cpu().numpy()

        r2 = r2_score(y_true, pred_np)
        rmse = np.sqrt(mean_squared_error(y_true, pred_np))
        mae = mean_absolute_error(y_true, pred_np)
        spearman, _ = spearmanr(y_true, pred_np)
        spearman = 0.0 if np.isnan(spearman) else spearman

        # 추가 메트릭 계산
        c_index = concordance_index(y_true, pred_np)
        kendall, _ = kendalltau(y_true, pred_np)
        kendall = 0.0 if np.isnan(kendall) else kendall
        cum_auc = cumulative_auc(y_true, pred_np)

        print(f"Patient {test_patient} Results:")
        print(f"  === Resistance-Assistance Continuum Performance ===")
        print(f"    R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, Spearman={spearman:.4f}")
        print(f"    C-index={c_index:.4f}, Kendall={kendall:.4f}, Cum_AUC={cum_auc:.4f}")

        # Task별 예측값 순서 검증
        print(f"  === Task-wise Continuum Order Validation ===")
        task_predictions = {}
        task_targets = {}

        for i, meta in enumerate(meta_test):
            task = meta['Task']
            domain = meta['Domain']

            if task not in task_predictions:
                task_predictions[task] = []
                task_targets[task] = []

            task_predictions[task].append(pred_np[i])
            task_targets[task].append(y_true[i])

        # Task별 평균 예측값 계산
        task_avg_pred = {}
        task_avg_target = {}
        for task in sorted(task_predictions.keys()):
            task_avg_pred[task] = np.mean(task_predictions[task])
            task_avg_target[task] = np.mean(task_targets[task])
            print(f"    Task {task}: Target={task_avg_target[task]:+.3f}, Predicted={task_avg_pred[task]:+.3f}")

        # Patient domain에 따른 예상 순서 검증
        patient_domain = meta_test[0]['Domain']  # 모든 테스트 데이터는 같은 환자
        if patient_domain == 'Patient':
            # Patient: Task3(Resistance:-1) < Task2(Compliant:0) < Task1(Stiff:+1)
            expected_order = [3, 2, 1]  # 오름차순 예상
            actual_order = sorted(task_avg_pred.keys(), key=lambda x: task_avg_pred[x])
            print(f"    Expected order: Task3 < Task2 < Task1 (Resistance < Compliant < Stiff)")
            print(f"    Actual order: {' < '.join([f'Task{t}' for t in actual_order])}")
            order_correct = (actual_order == expected_order)
            print(f"    Order Correctness: {'✓ CORRECT' if order_correct else '✗ INCORRECT'}")
        else:
            # Normal: Task2(Compliant:0) < Task1(Very Compliant:+0.5) < Task3(Stiff:+1)
            expected_order = [2, 1, 3]  # 오름차순 예상
            actual_order = sorted(task_avg_pred.keys(), key=lambda x: task_avg_pred[x])
            print(f"    Expected order: Task2 < Task1 < Task3 (Compliant < V.Compliant < Stiff)")
            print(f"    Actual order: {' < '.join([f'Task{t}' for t in actual_order])}")
            order_correct = (actual_order == expected_order)
            print(f"    Order Correctness: {'✓ CORRECT' if order_correct else '✗ INCORRECT'}")

        # 세그먼트 평균 단위 검증 (노이즈 억제 평가)
        print(f"  === Segment-Level Validation (Noise Suppression) ===")

        # (Number, Task, Repeat, State) 키별로 y_true, y_pred 수집
        segment_data = {}
        for i, meta in enumerate(meta_test if hasattr(meta_test, '__iter__') else []):
            key = (meta['Number'], meta['Task'], meta['Repeat'], meta['State'])
            if key not in segment_data:
                segment_data[key] = {'true': [], 'pred': []}

            segment_data[key]['true'].append(y_true[i])
            segment_data[key]['pred'].append(pred_np[i])

        # 세그먼트별 평균 계산 및 성능 평가
        segment_true = []
        segment_pred = []

        for key, data in segment_data.items():
            if len(data['true']) > 0:
                avg_true = np.mean(data['true'])
                avg_pred = np.mean(data['pred'])
                segment_true.append(avg_true)
                segment_pred.append(avg_pred)

        if len(segment_true) > 0:
            # 세그먼트 평균 기반 성능 계산
            seg_r2 = r2_score(segment_true, segment_pred)
            seg_rmse = np.sqrt(mean_squared_error(segment_true, segment_pred))
            seg_mae = mean_absolute_error(segment_true, segment_pred)
            seg_spearman, _ = spearmanr(segment_true, segment_pred)
            seg_spearman = 0.0 if np.isnan(seg_spearman) else seg_spearman

            # 세그먼트 레벨 추가 메트릭
            seg_c_index = concordance_index(np.array(segment_true), np.array(segment_pred))
            seg_kendall, _ = kendalltau(segment_true, segment_pred)
            seg_kendall = 0.0 if np.isnan(seg_kendall) else seg_kendall
            seg_cum_auc = cumulative_auc(np.array(segment_true), np.array(segment_pred))

            print(f"    Total segments: {len(segment_data)} (Number, Task, Repeat, State) combinations")
            print(f"    Segment-averaged performance:")
            print(f"      R²={seg_r2:.4f}, RMSE={seg_rmse:.4f}, MAE={seg_mae:.4f}, Spearman={seg_spearman:.4f}")
            print(f"      C-index={seg_c_index:.4f}, Kendall={seg_kendall:.4f}, Cum_AUC={seg_cum_auc:.4f}")

    return {
        'patient': test_patient,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'spearman': spearman,
        'c_index': c_index,
        'kendall': kendall,
        'cum_auc': cum_auc,
        'order_correct': order_correct,
        'task_avg_pred': task_avg_pred,
        'task_avg_target': task_avg_target,
        'predictions': pred_np,
        'targets': y_true
    }

def save_results_to_csv(results):
    """환자별 결과를 CSV 파일로 저장 (Ordinal Learning)"""
    # 환자별 상세 결과
    patient_results = []
    for result in results:
        patient_id = result['patient']
        row = {
            'Patient': patient_id,
            'R2': result['r2'],
            'RMSE': result['rmse'],
            'MAE': result['mae'],
            'Spearman': result['spearman'],
            'C_index': result['c_index'],
            'Kendall': result['kendall'],
            'Cum_AUC': result['cum_auc'],
            'Order_Correct': result['order_correct']
        }

        # Task별 예측값도 추가
        for task, pred_val in result['task_avg_pred'].items():
            row[f'Task{task}_Pred'] = pred_val
        for task, target_val in result['task_avg_target'].items():
            row[f'Task{task}_Target'] = target_val
        patient_results.append(row)

    # DataFrame 생성 및 저장
    df_results = pd.DataFrame(patient_results)
    df_results.to_csv('results/patient_results_ordinal_learning.csv', index=False)

    # 요약 통계
    r2_scores = [result['r2'] for result in results]
    rmse_scores = [result['rmse'] for result in results]
    mae_scores = [result['mae'] for result in results]
    spearman_scores = [result['spearman'] for result in results]
    c_index_scores = [result['c_index'] for result in results]
    kendall_scores = [result['kendall'] for result in results]
    cum_auc_scores = [result['cum_auc'] for result in results]

    summary_stats = [{
        'Metric': 'Resistance-Assistance Continuum',
        'R2_Mean': np.mean(r2_scores),
        'R2_Std': np.std(r2_scores),
        'RMSE_Mean': np.mean(rmse_scores),
        'RMSE_Std': np.std(rmse_scores),
        'MAE_Mean': np.mean(mae_scores),
        'MAE_Std': np.std(mae_scores),
        'Spearman_Mean': np.mean(spearman_scores),
        'Spearman_Std': np.std(spearman_scores),
        'C_index_Mean': np.mean(c_index_scores),
        'C_index_Std': np.std(c_index_scores),
        'Kendall_Mean': np.mean(kendall_scores),
        'Kendall_Std': np.std(kendall_scores),
        'Cum_AUC_Mean': np.mean(cum_auc_scores),
        'Cum_AUC_Std': np.std(cum_auc_scores)
    }]

    df_summary = pd.DataFrame(summary_stats)
    df_summary.to_csv('results/ordinal_learning_summary.csv', index=False)

    print(f"  Saved patient_results_ordinal_learning.csv ({len(patient_results)} patients)")
    print(f"  Saved ordinal_learning_summary.csv")

# 메인 실행
def main():
    print("Ordinal Learning for Resistance-Assistance Continuum with Transformer")

    # 데이터 로드
    print("Loading data...")
    df_normal, df_patient = load_data()
    df_combined = pd.concat([df_normal, df_patient])

    print(f"Normal subjects: {df_normal['Number'].nunique()}")
    print(f"Patient subjects: {df_patient['Number'].nunique()}")

    # MMT 정보 로드
    print("Loading MMT information...")
    mmt_dict = load_mmt_info()
    print(f"MMT grades: {mmt_dict}")

    # 라벨 생성
    print("Generating labels for Resistance-Assistance continuum...")
    all_stats = build_robust_stats(df_combined)
    labels_df = generate_labels(df_combined, all_stats, mmt_dict)
    print(f"Generated {len(labels_df)} label sequences")

    # LOSO 교차 검증 실행
    print("Starting LOSO Cross-Validation...")
    results = loso_cross_validation(df_normal, df_patient, labels_df, mmt_dict)

    # 결과 요약 (Ordinal Learning)
    print("\nOverall Results Summary:")
    if results:
        print(f"=== Resistance-Assistance Continuum Ordinal Learning Summary ===")

        # 전체 성능 계산
        r2_scores = [r['r2'] for r in results]
        rmse_scores = [r['rmse'] for r in results]
        mae_scores = [r['mae'] for r in results]
        spearman_scores = [r['spearman'] for r in results]
        c_index_scores = [r['c_index'] for r in results]
        kendall_scores = [r['kendall'] for r in results]
        cum_auc_scores = [r['cum_auc'] for r in results]

        print(f"  Resistance-Assistance Continuum Performance:")
        print(f"    R² - Mean: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
        print(f"    RMSE - Mean: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
        print(f"    MAE - Mean: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
        print(f"    Spearman - Mean: {np.mean(spearman_scores):.4f} ± {np.std(spearman_scores):.4f}")
        print(f"    C-index - Mean: {np.mean(c_index_scores):.4f} ± {np.std(c_index_scores):.4f}")
        print(f"    Kendall - Mean: {np.mean(kendall_scores):.4f} ± {np.std(kendall_scores):.4f}")
        print(f"    Cum_AUC - Mean: {np.mean(cum_auc_scores):.4f} ± {np.std(cum_auc_scores):.4f}")

        # 순서 정확도 통계
        order_accuracy = [r['order_correct'] for r in results]
        correct_count = sum(order_accuracy)
        total_count = len(order_accuracy)
        print(f"    Task Order Accuracy: {correct_count}/{total_count} ({100*correct_count/total_count:.1f}%)")

        # 환자별 성능 순위
        print(f"\n=== Patient Performance Ranking (by R²) ===")
        sorted_patients = sorted(results, key=lambda x: x['r2'], reverse=True)
        for i, result in enumerate(sorted_patients):
            order_status = "✓" if result['order_correct'] else "✗"
            print(f"  {i+1}. Patient {result['patient']}: R²={result['r2']:.4f} [{order_status}]")

        # 순서 정확도별 분석
        print(f"\n=== Task Order Analysis ===")
        correct_patients = [r for r in results if r['order_correct']]
        incorrect_patients = [r for r in results if not r['order_correct']]

        if correct_patients:
            print(f"  Correct Order Patients ({len(correct_patients)}): {[r['patient'] for r in correct_patients]}")
            print(f"    Average R²: {np.mean([r['r2'] for r in correct_patients]):.4f}")

        if incorrect_patients:
            print(f"  Incorrect Order Patients ({len(incorrect_patients)}): {[r['patient'] for r in incorrect_patients]}")
            print(f"    Average R²: {np.mean([r['r2'] for r in incorrect_patients]):.4f}")

            # 잘못된 순서의 환자들 상세 분석
            print(f"  Detailed Analysis of Incorrect Cases:")
            for result in incorrect_patients:
                patient_id = result['patient']
                pred_order = sorted(result['task_avg_pred'].keys(), key=lambda x: result['task_avg_pred'][x])
                print(f"    Patient {patient_id}: {' < '.join([f'Task{t}({result['task_avg_pred'][t]:+.3f})' for t in pred_order])}")

        # CSV 파일로 환자별 결과 저장
        print(f"\nSaving results to CSV...")
        save_results_to_csv(results)

    print("\nTraining completed!")

if __name__ == "__main__":
    main()