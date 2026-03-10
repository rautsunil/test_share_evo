"""
Customer Tower Evaluation Framework

Comprehensive evaluation of Customer Tower embeddings including:
1. Intrinsic Metrics (embedding quality)
2. Linear Probe Evaluation (representation usefulness)
3. Downstream Task Performance (actual CRM tasks)
4. Ablation Studies (contribution analysis)
5. Benchmark Comparisons (vs XGBoost, MLP, etc.)

Author: EvoCRM Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    classification_report, confusion_matrix
)
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, kendalltau
import warnings
warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class EvaluationConfig:
    """Configuration for Customer Tower evaluation."""
    
    # Train/Val/Test split
    test_size: float = 0.2
    val_size: float = 0.1
    
    # Cross-validation
    n_folds: int = 5
    
    # Clustering evaluation
    n_clusters_range: List[int] = field(default_factory=lambda: [3, 5, 8, 10, 15])
    
    # Linear probe
    probe_max_iter: int = 1000
    
    # Visualization
    tsne_perplexity: int = 30
    
    # Random seed
    seed: int = 42


# ============================================================
# LEVEL 1: INTRINSIC EVALUATION
# ============================================================

class IntrinsicEvaluator:
    """
    Evaluate embedding quality without downstream tasks.
    
    Metrics:
    - Embedding statistics (mean, std, norms)
    - Isotropy (how uniformly distributed in space)
    - Effective rank (dimensionality utilization)
    - Reconstruction quality (if applicable)
    """
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
    
    def evaluate(
        self,
        embeddings: np.ndarray,
        original_features: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Run all intrinsic evaluations.
        
        Args:
            embeddings: Customer Tower output (N, embedding_dim)
            original_features: Original input features (for reconstruction)
            labels: Optional labels (for class separation analysis)
        
        Returns:
            Dictionary of metrics
        """
        results = {}
        
        # Basic statistics
        results['basic_stats'] = self._compute_basic_stats(embeddings)
        
        # Isotropy
        results['isotropy'] = self._compute_isotropy(embeddings)
        
        # Effective rank
        results['effective_rank'] = self._compute_effective_rank(embeddings)
        
        # Uniformity and alignment (if labels provided)
        if labels is not None:
            results['uniformity'] = self._compute_uniformity(embeddings)
            results['alignment'] = self._compute_alignment(embeddings, labels)
        
        # Clustering quality (unsupervised)
        results['clustering'] = self._evaluate_clustering(embeddings)
        
        return results
    
    def _compute_basic_stats(self, embeddings: np.ndarray) -> Dict:
        """Compute basic embedding statistics."""
        norms = np.linalg.norm(embeddings, axis=1)
        
        return {
            'mean': float(embeddings.mean()),
            'std': float(embeddings.std()),
            'min': float(embeddings.min()),
            'max': float(embeddings.max()),
            'norm_mean': float(norms.mean()),
            'norm_std': float(norms.std()),
            'dimension': embeddings.shape[1],
            'num_samples': embeddings.shape[0],
        }
    
    def _compute_isotropy(self, embeddings: np.ndarray) -> Dict:
        """
        Compute isotropy of embeddings.
        
        Isotropy measures how uniformly the embeddings are distributed
        in the embedding space. Higher is better (more diverse).
        
        Reference: "On the Sentence Embeddings from Pre-trained Language Models"
        """
        # Compute covariance matrix
        centered = embeddings - embeddings.mean(axis=0)
        cov = np.cov(centered.T)
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-10)  # Avoid log(0)
        
        # Isotropy score (based on eigenvalue distribution)
        # Perfectly isotropic = all eigenvalues equal
        normalized_eig = eigenvalues / eigenvalues.sum()
        entropy = -np.sum(normalized_eig * np.log(normalized_eig + 1e-10))
        max_entropy = np.log(len(eigenvalues))
        
        isotropy_score = entropy / max_entropy if max_entropy > 0 else 0
        
        # Condition number (lower is better)
        condition_number = eigenvalues.max() / (eigenvalues.min() + 1e-10)
        
        return {
            'isotropy_score': float(isotropy_score),
            'condition_number': float(condition_number),
            'top_eigenvalue_ratio': float(eigenvalues[-1] / eigenvalues.sum()),
        }
    
    def _compute_effective_rank(self, embeddings: np.ndarray) -> Dict:
        """
        Compute effective rank of embedding matrix.
        
        Measures how many dimensions are actually being used.
        Higher effective rank = better utilization of embedding space.
        """
        # SVD
        _, s, _ = np.linalg.svd(embeddings, full_matrices=False)
        
        # Effective rank (based on entropy of singular values)
        normalized_s = s / s.sum()
        entropy = -np.sum(normalized_s * np.log(normalized_s + 1e-10))
        effective_rank = np.exp(entropy)
        
        # Percentage of variance explained by top-k components
        variance_explained = np.cumsum(s**2) / np.sum(s**2)
        
        return {
            'effective_rank': float(effective_rank),
            'rank_ratio': float(effective_rank / len(s)),
            'variance_top_10': float(variance_explained[min(9, len(s)-1)]),
            'variance_top_50': float(variance_explained[min(49, len(s)-1)]),
            '95_variance_dims': int(np.argmax(variance_explained >= 0.95) + 1),
        }
    
    def _compute_uniformity(self, embeddings: np.ndarray) -> Dict:
        """
        Compute uniformity of embeddings.
        
        Uniformity measures how well the embeddings are spread out.
        Lower uniformity loss = better spread.
        
        Reference: "Understanding Contrastive Representation Learning"
        """
        # Normalize embeddings
        normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Sample pairs for efficiency
        n = min(5000, len(embeddings))
        indices = np.random.choice(len(embeddings), n, replace=False)
        sampled = normalized[indices]
        
        # Compute pairwise squared distances
        sq_dists = cdist(sampled, sampled, 'sqeuclidean')
        
        # Uniformity loss (log of average Gaussian kernel)
        t = 2  # Temperature
        uniformity_loss = np.log(np.exp(-t * sq_dists).mean())
        
        return {
            'uniformity_loss': float(uniformity_loss),
            'avg_pairwise_distance': float(np.sqrt(sq_dists).mean()),
        }
    
    def _compute_alignment(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray
    ) -> Dict:
        """
        Compute alignment of embeddings with labels.
        
        Alignment measures how close same-class samples are.
        Lower alignment loss = better class coherence.
        """
        # Normalize embeddings
        normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Compute class-wise statistics
        unique_labels = np.unique(labels)
        
        intra_class_dists = []
        inter_class_dists = []
        
        for label in unique_labels:
            mask = labels == label
            class_embeddings = normalized[mask]
            
            if len(class_embeddings) > 1:
                # Intra-class distance
                intra_dist = cdist(class_embeddings, class_embeddings, 'euclidean')
                intra_class_dists.append(intra_dist[np.triu_indices(len(class_embeddings), k=1)].mean())
                
                # Inter-class distance (vs other classes)
                other_embeddings = normalized[~mask]
                if len(other_embeddings) > 0:
                    inter_dist = cdist(class_embeddings, other_embeddings, 'euclidean')
                    inter_class_dists.append(inter_dist.mean())
        
        avg_intra = np.mean(intra_class_dists) if intra_class_dists else 0
        avg_inter = np.mean(inter_class_dists) if inter_class_dists else 0
        
        # Separation ratio (higher is better)
        separation_ratio = avg_inter / (avg_intra + 1e-10)
        
        return {
            'alignment_loss': float(avg_intra),
            'avg_intra_class_distance': float(avg_intra),
            'avg_inter_class_distance': float(avg_inter),
            'separation_ratio': float(separation_ratio),
        }
    
    def _evaluate_clustering(self, embeddings: np.ndarray) -> Dict:
        """Evaluate clustering quality of embeddings."""
        results = {}
        
        for k in self.config.n_clusters_range:
            if k >= len(embeddings):
                continue
            
            kmeans = KMeans(n_clusters=k, random_state=self.config.seed, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            results[f'k={k}'] = {
                'silhouette': float(silhouette_score(embeddings, cluster_labels)),
                'calinski_harabasz': float(calinski_harabasz_score(embeddings, cluster_labels)),
                'davies_bouldin': float(davies_bouldin_score(embeddings, cluster_labels)),
                'inertia': float(kmeans.inertia_),
            }
        
        return results


# ============================================================
# LEVEL 2: LINEAR PROBE EVALUATION
# ============================================================

class LinearProbeEvaluator:
    """
    Evaluate embeddings using linear probes.
    
    A good embedding should enable simple models to perform well.
    This tests whether the embedding captures useful information.
    
    Tests:
    - Linear classifier for churn prediction
    - Linear regressor for CLV estimation
    - k-NN classifier
    """
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
    
    def evaluate_classification(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        task_name: str = "classification"
    ) -> Dict[str, Any]:
        """
        Evaluate embeddings for classification task using linear probe.
        
        Args:
            embeddings: Customer Tower embeddings (N, dim)
            labels: Binary or multi-class labels
            task_name: Name of the task
        
        Returns:
            Dictionary with classification metrics
        """
        results = {'task': task_name}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels,
            test_size=self.config.test_size,
            random_state=self.config.seed,
            stratify=labels if len(np.unique(labels)) > 1 else None
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Logistic Regression (Linear Probe)
        lr = LogisticRegression(
            max_iter=self.config.probe_max_iter,
            random_state=self.config.seed
        )
        lr.fit(X_train_scaled, y_train)
        y_pred_lr = lr.predict(X_test_scaled)
        y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1] if len(np.unique(labels)) == 2 else None
        
        results['linear_probe'] = {
            'accuracy': float(accuracy_score(y_test, y_pred_lr)),
            'precision': float(precision_score(y_test, y_pred_lr, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred_lr, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_test, y_pred_lr, average='weighted', zero_division=0)),
        }
        
        if y_prob_lr is not None:
            results['linear_probe']['auc_roc'] = float(roc_auc_score(y_test, y_prob_lr))
        
        # 2. k-NN (tests local structure)
        for k in [1, 3, 5, 10]:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_scaled, y_train)
            y_pred_knn = knn.predict(X_test_scaled)
            
            results[f'knn_k={k}'] = {
                'accuracy': float(accuracy_score(y_test, y_pred_knn)),
                'f1': float(f1_score(y_test, y_pred_knn, average='weighted', zero_division=0)),
            }
        
        # 3. Cross-validation
        cv = StratifiedKFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.seed)
        cv_scores = cross_val_score(lr, embeddings, labels, cv=cv, scoring='accuracy')
        
        results['cross_validation'] = {
            'mean_accuracy': float(cv_scores.mean()),
            'std_accuracy': float(cv_scores.std()),
            'scores': cv_scores.tolist(),
        }
        
        return results
    
    def evaluate_regression(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        task_name: str = "regression"
    ) -> Dict[str, Any]:
        """
        Evaluate embeddings for regression task using linear probe.
        
        Args:
            embeddings: Customer Tower embeddings
            targets: Continuous targets (e.g., CLV)
            task_name: Name of the task
        
        Returns:
            Dictionary with regression metrics
        """
        results = {'task': task_name}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, targets,
            test_size=self.config.test_size,
            random_state=self.config.seed
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Ridge Regression (Linear Probe)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)
        y_pred_ridge = ridge.predict(X_test_scaled)
        
        results['linear_probe'] = {
            'mse': float(mean_squared_error(y_test, y_pred_ridge)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_ridge))),
            'mae': float(mean_absolute_error(y_test, y_pred_ridge)),
            'r2': float(r2_score(y_test, y_pred_ridge)),
        }
        
        # 2. Correlation
        spearman_corr, _ = spearmanr(y_test, y_pred_ridge)
        kendall_corr, _ = kendalltau(y_test, y_pred_ridge)
        
        results['correlation'] = {
            'spearman': float(spearman_corr),
            'kendall': float(kendall_corr),
        }
        
        # 3. Cross-validation
        cv_scores = cross_val_score(
            ridge, embeddings, targets,
            cv=self.config.n_folds,
            scoring='r2'
        )
        
        results['cross_validation'] = {
            'mean_r2': float(cv_scores.mean()),
            'std_r2': float(cv_scores.std()),
        }
        
        return results


# ============================================================
# LEVEL 3: DOWNSTREAM TASK EVALUATION
# ============================================================

class DownstreamTaskEvaluator:
    """
    Evaluate on actual CRM downstream tasks.
    
    Compares:
    - Customer Tower embeddings + simple head
    - Traditional ML on raw features
    """
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
    
    def evaluate_churn_prediction(
        self,
        embeddings: np.ndarray,
        raw_features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate churn prediction task.
        
        Compares:
        1. Embedding + Logistic Regression
        2. XGBoost on raw features (benchmark)
        3. Random Forest on raw features (benchmark)
        """
        results = {}
        
        # Split
        X_emb_train, X_emb_test, X_raw_train, X_raw_test, y_train, y_test = train_test_split(
            embeddings, raw_features, labels,
            test_size=self.config.test_size,
            random_state=self.config.seed,
            stratify=labels
        )
        
        # 1. Embedding + Logistic Regression
        scaler = StandardScaler()
        X_emb_train_scaled = scaler.fit_transform(X_emb_train)
        X_emb_test_scaled = scaler.transform(X_emb_test)
        
        lr = LogisticRegression(max_iter=self.config.probe_max_iter, random_state=self.config.seed)
        lr.fit(X_emb_train_scaled, y_train)
        y_prob_emb = lr.predict_proba(X_emb_test_scaled)[:, 1]
        y_pred_emb = (y_prob_emb > 0.5).astype(int)
        
        results['embedding_linear'] = {
            'accuracy': float(accuracy_score(y_test, y_pred_emb)),
            'precision': float(precision_score(y_test, y_pred_emb, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred_emb, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred_emb, zero_division=0)),
            'auc_roc': float(roc_auc_score(y_test, y_prob_emb)),
        }
        
        # 2. XGBoost on raw features (strong benchmark)
        if HAS_XGB:
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config.seed,
                eval_metric='logloss'
            )
            xgb_model.fit(X_raw_train, y_train)
            y_prob_xgb = xgb_model.predict_proba(X_raw_test)[:, 1]
            y_pred_xgb = (y_prob_xgb > 0.5).astype(int)
            
            results['xgboost_raw'] = {
                'accuracy': float(accuracy_score(y_test, y_pred_xgb)),
                'precision': float(precision_score(y_test, y_pred_xgb, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred_xgb, zero_division=0)),
                'f1': float(f1_score(y_test, y_pred_xgb, zero_division=0)),
                'auc_roc': float(roc_auc_score(y_test, y_prob_xgb)),
            }
        
        # 3. Random Forest on raw features
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.config.seed
        )
        rf.fit(X_raw_train, y_train)
        y_prob_rf = rf.predict_proba(X_raw_test)[:, 1]
        y_pred_rf = (y_prob_rf > 0.5).astype(int)
        
        results['random_forest_raw'] = {
            'accuracy': float(accuracy_score(y_test, y_pred_rf)),
            'precision': float(precision_score(y_test, y_pred_rf, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred_rf, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred_rf, zero_division=0)),
            'auc_roc': float(roc_auc_score(y_test, y_prob_rf)),
        }
        
        # 4. Random embeddings (sanity check)
        random_emb = np.random.randn(*embeddings.shape)
        X_rand_train, X_rand_test = train_test_split(
            random_emb, test_size=self.config.test_size, random_state=self.config.seed
        )
        
        lr_rand = LogisticRegression(max_iter=self.config.probe_max_iter, random_state=self.config.seed)
        lr_rand.fit(X_rand_train, y_train)
        y_prob_rand = lr_rand.predict_proba(X_rand_test)[:, 1]
        
        results['random_embedding'] = {
            'auc_roc': float(roc_auc_score(y_test, y_prob_rand)),
        }
        
        # Summary
        results['summary'] = {
            'embedding_vs_xgboost_auc_diff': results['embedding_linear']['auc_roc'] - results.get('xgboost_raw', {}).get('auc_roc', 0),
            'embedding_vs_random_auc_diff': results['embedding_linear']['auc_roc'] - results['random_embedding']['auc_roc'],
        }
        
        return results
    
    def evaluate_clv_prediction(
        self,
        embeddings: np.ndarray,
        raw_features: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate CLV prediction task.
        
        Similar structure to churn but for regression.
        """
        results = {}
        
        # Log-transform CLV (common practice)
        targets_log = np.log1p(targets)
        
        # Split
        X_emb_train, X_emb_test, X_raw_train, X_raw_test, y_train, y_test = train_test_split(
            embeddings, raw_features, targets_log,
            test_size=self.config.test_size,
            random_state=self.config.seed
        )
        
        # 1. Embedding + Ridge
        scaler = StandardScaler()
        X_emb_train_scaled = scaler.fit_transform(X_emb_train)
        X_emb_test_scaled = scaler.transform(X_emb_test)
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_emb_train_scaled, y_train)
        y_pred_emb = ridge.predict(X_emb_test_scaled)
        
        results['embedding_linear'] = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_emb))),
            'mae': float(mean_absolute_error(y_test, y_pred_emb)),
            'r2': float(r2_score(y_test, y_pred_emb)),
        }
        
        # 2. XGBoost on raw features
        if HAS_XGB:
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config.seed
            )
            xgb_model.fit(X_raw_train, y_train)
            y_pred_xgb = xgb_model.predict(X_raw_test)
            
            results['xgboost_raw'] = {
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_xgb))),
                'mae': float(mean_absolute_error(y_test, y_pred_xgb)),
                'r2': float(r2_score(y_test, y_pred_xgb)),
            }
        
        # 3. Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=self.config.seed)
        # Note: Using regressor
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=self.config.seed)
        rf.fit(X_raw_train, y_train)
        y_pred_rf = rf.predict(X_raw_test)
        
        results['random_forest_raw'] = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_rf))),
            'mae': float(mean_absolute_error(y_test, y_pred_rf)),
            'r2': float(r2_score(y_test, y_pred_rf)),
        }
        
        return results


# ============================================================
# LEVEL 4: ABLATION STUDIES
# ============================================================

class AblationStudyEvaluator:
    """
    Ablation studies to understand Customer Tower contribution.
    
    Tests:
    1. Full model vs No Customer Tower
    2. Customer Tower only vs Full Model
    3. Feature group importance
    """
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
    
    def evaluate_feature_groups(
        self,
        embeddings: np.ndarray,
        feature_groups: Dict[str, np.ndarray],
        labels: np.ndarray,
        task: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Evaluate contribution of different feature groups.
        
        Args:
            embeddings: Full Customer Tower embeddings
            feature_groups: Dict mapping group name to feature array
                e.g., {'demographics': demo_features, 'behavioral': behav_features}
            labels: Target labels
            task: 'classification' or 'regression'
        
        Returns:
            Performance metrics for each feature group
        """
        results = {}
        
        # Evaluate full embeddings
        if task == 'classification':
            full_metrics = self._evaluate_classification(embeddings, labels)
            results['full_embedding'] = full_metrics
        else:
            full_metrics = self._evaluate_regression(embeddings, labels)
            results['full_embedding'] = full_metrics
        
        # Evaluate each feature group
        for group_name, features in feature_groups.items():
            if task == 'classification':
                group_metrics = self._evaluate_classification(features, labels)
            else:
                group_metrics = self._evaluate_regression(features, labels)
            
            results[group_name] = group_metrics
            
            # Calculate contribution
            if task == 'classification':
                results[f'{group_name}_contribution'] = {
                    'auc_diff_from_full': full_metrics.get('auc_roc', 0) - group_metrics.get('auc_roc', 0),
                }
            else:
                results[f'{group_name}_contribution'] = {
                    'r2_diff_from_full': full_metrics.get('r2', 0) - group_metrics.get('r2', 0),
                }
        
        return results
    
    def _evaluate_classification(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """Helper for classification evaluation."""
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=self.config.test_size,
            random_state=self.config.seed,
            stratify=labels
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr = LogisticRegression(max_iter=self.config.probe_max_iter, random_state=self.config.seed)
        lr.fit(X_train_scaled, y_train)
        
        y_pred = lr.predict(X_test_scaled)
        y_prob = lr.predict_proba(X_test_scaled)[:, 1]
        
        return {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'auc_roc': float(roc_auc_score(y_test, y_prob)),
        }
    
    def _evaluate_regression(self, features: np.ndarray, targets: np.ndarray) -> Dict:
        """Helper for regression evaluation."""
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets,
            test_size=self.config.test_size,
            random_state=self.config.seed
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)
        y_pred = ridge.predict(X_test_scaled)
        
        return {
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'r2': float(r2_score(y_test, y_pred)),
        }


# ============================================================
# BENCHMARK THRESHOLDS
# ============================================================

BENCHMARK_THRESHOLDS = {
    'churn_prediction': {
        'poor': {'auc_roc': 0.60, 'f1': 0.40},
        'acceptable': {'auc_roc': 0.70, 'f1': 0.50},
        'good': {'auc_roc': 0.80, 'f1': 0.60},
        'excellent': {'auc_roc': 0.85, 'f1': 0.70},
        'state_of_art': {'auc_roc': 0.90, 'f1': 0.80},
    },
    'clv_prediction': {
        'poor': {'r2': 0.20, 'spearman': 0.30},
        'acceptable': {'r2': 0.40, 'spearman': 0.50},
        'good': {'r2': 0.60, 'spearman': 0.70},
        'excellent': {'r2': 0.75, 'spearman': 0.80},
        'state_of_art': {'r2': 0.85, 'spearman': 0.90},
    },
    'embedding_quality': {
        'poor': {'isotropy': 0.30, 'effective_rank_ratio': 0.20},
        'acceptable': {'isotropy': 0.50, 'effective_rank_ratio': 0.40},
        'good': {'isotropy': 0.70, 'effective_rank_ratio': 0.60},
        'excellent': {'isotropy': 0.85, 'effective_rank_ratio': 0.75},
    },
    'linear_probe_vs_xgboost': {
        'description': 'AUC difference between embedding+linear vs XGBoost on raw features',
        'poor': -0.15,      # >15% worse than XGBoost
        'acceptable': -0.05, # 5% worse than XGBoost
        'good': 0.0,         # On par with XGBoost
        'excellent': 0.05,   # 5% better than XGBoost
    }
}


# ============================================================
# MAIN EVALUATION CLASS
# ============================================================

class CustomerTowerEvaluator:
    """
    Comprehensive Customer Tower evaluation.
    
    Combines all evaluation levels into a single interface.
    """
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        
        self.intrinsic = IntrinsicEvaluator(self.config)
        self.linear_probe = LinearProbeEvaluator(self.config)
        self.downstream = DownstreamTaskEvaluator(self.config)
        self.ablation = AblationStudyEvaluator(self.config)
    
    def evaluate_all(
        self,
        embeddings: np.ndarray,
        raw_features: np.ndarray,
        churn_labels: np.ndarray,
        clv_targets: np.ndarray,
        feature_groups: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation.
        
        Args:
            embeddings: Customer Tower embeddings (N, dim)
            raw_features: Original input features (N, num_features)
            churn_labels: Binary churn labels
            clv_targets: Continuous CLV values
            feature_groups: Optional dict of feature groups for ablation
        
        Returns:
            Comprehensive evaluation results
        """
        results = {}
        
        print("="*60)
        print("CUSTOMER TOWER COMPREHENSIVE EVALUATION")
        print("="*60)
        
        # Level 1: Intrinsic
        print("\n[1/4] Running Intrinsic Evaluation...")
        results['intrinsic'] = self.intrinsic.evaluate(
            embeddings, raw_features, churn_labels
        )
        
        # Level 2: Linear Probe
        print("[2/4] Running Linear Probe Evaluation...")
        results['linear_probe_churn'] = self.linear_probe.evaluate_classification(
            embeddings, churn_labels, "churn_prediction"
        )
        results['linear_probe_clv'] = self.linear_probe.evaluate_regression(
            embeddings, clv_targets, "clv_prediction"
        )
        
        # Level 3: Downstream Tasks
        print("[3/4] Running Downstream Task Evaluation...")
        results['downstream_churn'] = self.downstream.evaluate_churn_prediction(
            embeddings, raw_features, churn_labels
        )
        results['downstream_clv'] = self.downstream.evaluate_clv_prediction(
            embeddings, raw_features, clv_targets
        )
        
        # Level 4: Ablation
        if feature_groups:
            print("[4/4] Running Ablation Studies...")
            results['ablation'] = self.ablation.evaluate_feature_groups(
                embeddings, feature_groups, churn_labels, task='classification'
            )
        else:
            print("[4/4] Skipping Ablation (no feature groups provided)")
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate evaluation summary with grades."""
        summary = {}
        
        # Intrinsic quality
        isotropy = results['intrinsic']['isotropy']['isotropy_score']
        eff_rank_ratio = results['intrinsic']['effective_rank']['rank_ratio']
        
        summary['embedding_quality'] = {
            'isotropy': isotropy,
            'effective_rank_ratio': eff_rank_ratio,
            'grade': self._get_grade('embedding_quality', {
                'isotropy': isotropy,
                'effective_rank_ratio': eff_rank_ratio
            }),
        }
        
        # Churn prediction
        churn_auc = results['downstream_churn']['embedding_linear']['auc_roc']
        churn_f1 = results['downstream_churn']['embedding_linear']['f1']
        
        summary['churn_prediction'] = {
            'auc_roc': churn_auc,
            'f1': churn_f1,
            'grade': self._get_grade('churn_prediction', {
                'auc_roc': churn_auc, 'f1': churn_f1
            }),
        }
        
        # CLV prediction
        clv_r2 = results['downstream_clv']['embedding_linear']['r2']
        
        summary['clv_prediction'] = {
            'r2': clv_r2,
            'grade': self._get_grade('clv_prediction', {'r2': clv_r2}),
        }
        
        # Comparison with XGBoost
        if 'xgboost_raw' in results['downstream_churn']:
            xgb_auc = results['downstream_churn']['xgboost_raw']['auc_roc']
            summary['vs_xgboost'] = {
                'embedding_auc': churn_auc,
                'xgboost_auc': xgb_auc,
                'difference': churn_auc - xgb_auc,
                'verdict': 'BETTER' if churn_auc > xgb_auc else 'WORSE' if churn_auc < xgb_auc else 'EQUAL',
            }
        
        # Sanity check (vs random)
        random_auc = results['downstream_churn']['random_embedding']['auc_roc']
        summary['sanity_check'] = {
            'random_embedding_auc': random_auc,
            'improvement_over_random': churn_auc - random_auc,
            'passed': churn_auc > random_auc + 0.05,  # At least 5% better
        }
        
        return summary
    
    def _get_grade(self, metric_type: str, values: Dict) -> str:
        """Get grade based on benchmark thresholds."""
        thresholds = BENCHMARK_THRESHOLDS.get(metric_type, {})
        
        for grade in ['state_of_art', 'excellent', 'good', 'acceptable', 'poor']:
            if grade not in thresholds:
                continue
            
            threshold = thresholds[grade]
            
            if isinstance(threshold, dict):
                if all(values.get(k, 0) >= v for k, v in threshold.items()):
                    return grade.upper()
            else:
                if all(v >= threshold for v in values.values()):
                    return grade.upper()
        
        return 'POOR'
    
    def print_report(self, results: Dict):
        """Print formatted evaluation report."""
        summary = results.get('summary', {})
        
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        
        # Embedding Quality
        eq = summary.get('embedding_quality', {})
        print(f"\n📊 EMBEDDING QUALITY: {eq.get('grade', 'N/A')}")
        print(f"   Isotropy Score: {eq.get('isotropy', 0):.3f}")
        print(f"   Effective Rank Ratio: {eq.get('effective_rank_ratio', 0):.3f}")
        
        # Churn Prediction
        cp = summary.get('churn_prediction', {})
        print(f"\n🔮 CHURN PREDICTION: {cp.get('grade', 'N/A')}")
        print(f"   AUC-ROC: {cp.get('auc_roc', 0):.3f}")
        print(f"   F1 Score: {cp.get('f1', 0):.3f}")
        
        # CLV Prediction
        clv = summary.get('clv_prediction', {})
        print(f"\n💰 CLV PREDICTION: {clv.get('grade', 'N/A')}")
        print(f"   R² Score: {clv.get('r2', 0):.3f}")
        
        # Comparison with XGBoost
        vs_xgb = summary.get('vs_xgboost', {})
        if vs_xgb:
            print(f"\n🆚 VS XGBOOST BENCHMARK:")
            print(f"   Embedding AUC: {vs_xgb.get('embedding_auc', 0):.3f}")
            print(f"   XGBoost AUC: {vs_xgb.get('xgboost_auc', 0):.3f}")
            print(f"   Difference: {vs_xgb.get('difference', 0):+.3f}")
            print(f"   Verdict: {vs_xgb.get('verdict', 'N/A')}")
        
        # Sanity Check
        sanity = summary.get('sanity_check', {})
        print(f"\n✅ SANITY CHECK: {'PASSED' if sanity.get('passed', False) else 'FAILED'}")
        print(f"   Improvement over random: {sanity.get('improvement_over_random', 0):.3f}")
        
        # Overall Assessment
        print("\n" + "="*60)
        print("OVERALL ASSESSMENT")
        print("="*60)
        
        grades = [
            eq.get('grade', 'POOR'),
            cp.get('grade', 'POOR'),
            clv.get('grade', 'POOR'),
        ]
        
        grade_scores = {'STATE_OF_ART': 5, 'EXCELLENT': 4, 'GOOD': 3, 'ACCEPTABLE': 2, 'POOR': 1}
        avg_score = np.mean([grade_scores.get(g, 1) for g in grades])
        
        if avg_score >= 4.5:
            overall = "EXCELLENT - Customer Tower is performing very well!"
        elif avg_score >= 3.5:
            overall = "GOOD - Customer Tower is performing well."
        elif avg_score >= 2.5:
            overall = "ACCEPTABLE - Customer Tower needs improvement."
        else:
            overall = "POOR - Customer Tower needs significant improvement."
        
        print(f"\n{overall}")
        
        # Recommendations
        print("\n📝 RECOMMENDATIONS:")
        
        if eq.get('isotropy', 0) < 0.5:
            print("   • Low isotropy: Consider adding dropout or using contrastive learning")
        
        if eq.get('effective_rank_ratio', 0) < 0.4:
            print("   • Low effective rank: Embedding dimensions may be underutilized")
        
        if vs_xgb.get('difference', 0) < -0.05:
            print("   • Underperforming vs XGBoost: Consider increasing model capacity or training longer")
        
        if not sanity.get('passed', False):
            print("   • CRITICAL: Embeddings not much better than random - check training!")


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    # Simulate data
    n_samples = 5000
    embedding_dim = 256
    n_features = 30
    
    # Simulated embeddings (pretend these came from Customer Tower)
    embeddings = np.random.randn(n_samples, embedding_dim)
    
    # Add some structure (clusters)
    labels = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 20% churn
    embeddings[labels == 1] += 0.5  # Churned users slightly different
    
    # Raw features
    raw_features = np.random.randn(n_samples, n_features)
    
    # CLV targets
    clv_targets = np.exp(np.random.randn(n_samples) + 8)  # Log-normal CLV
    
    # Run evaluation
    evaluator = CustomerTowerEvaluator()
    
    results = evaluator.evaluate_all(
        embeddings=embeddings,
        raw_features=raw_features,
        churn_labels=labels,
        clv_targets=clv_targets
    )
    
    # Print report
    evaluator.print_report(results)
