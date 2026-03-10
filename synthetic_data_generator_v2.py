"""
Synthetic Data Generator for EvoCRM

Generates synthetic data matching your exact table structure:
1. Demographics Table - User static attributes
2. Transactions Table - Purchase history
3. Web Behavior Table - Product views, clicks, cart actions
4. Campaigns Table - Campaign interactions (SPARSE)

Features:
- Realistic correlations between tables
- Configurable distributions
- Handles sparse campaign data
- Generates proper target variables
- Ready for EvoCRM training

Author: EvoCRM Team
Version: 2.0.0
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import hashlib
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class DataGeneratorConfig:
    """
    Configuration for synthetic data generation.
    
    Customize these parameters to match your real data distributions.
    """
    
    # === VOLUME SETTINGS ===
    num_users: int = 100_000
    num_products: int = 2_000
    
    # Average transactions per user (varies by segment)
    avg_transactions_per_user: float = 8.0
    
    # Average web events per user (varies by segment)
    avg_web_events_per_user: float = 35.0
    
    # Campaign coverage (what % of users have campaign data)
    campaign_coverage: float = 0.15  # 15% - SPARSE!
    avg_campaigns_per_exposed_user: float = 3.0
    
    # === TIME SETTINGS ===
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    
    # === DEMOGRAPHIC DISTRIBUTIONS ===
    cities: Dict[str, float] = field(default_factory=lambda: {
        'Mumbai': 0.18,
        'Delhi': 0.15,
        'Bangalore': 0.14,
        'Chennai': 0.10,
        'Hyderabad': 0.10,
        'Kolkata': 0.08,
        'Pune': 0.07,
        'Ahmedabad': 0.06,
        'Jaipur': 0.05,
        'Lucknow': 0.04,
        'Other': 0.03,
    })
    
    age_groups: Dict[str, float] = field(default_factory=lambda: {
        '18-24': 0.18,
        '25-34': 0.32,
        '35-44': 0.25,
        '45-54': 0.15,
        '55+': 0.10,
    })
    
    genders: Dict[str, float] = field(default_factory=lambda: {
        'M': 0.48,
        'F': 0.48,
        'Other': 0.04,
    })
    
    # === USER SEGMENTS (Hidden - drives behavior) ===
    user_segments: Dict[str, float] = field(default_factory=lambda: {
        'High Value': 0.10,      # VIPs, frequent buyers
        'Regular': 0.25,         # Consistent buyers
        'Occasional': 0.30,      # Sometimes buy
        'Window Shopper': 0.20,  # Browse but rarely buy
        'Churned': 0.15,         # Were active, now inactive
    })
    
    # === PRODUCT CATEGORIES ===
    product_categories: Dict[str, float] = field(default_factory=lambda: {
        'Electronics': 0.15,
        'Fashion': 0.25,
        'Home & Living': 0.15,
        'Beauty': 0.12,
        'Sports': 0.08,
        'Books': 0.08,
        'Food & Grocery': 0.10,
        'Others': 0.07,
    })
    
    # === WEB EVENT TYPES ===
    web_event_types: Dict[str, float] = field(default_factory=lambda: {
        'product_view': 0.45,
        'category_view': 0.15,
        'search': 0.12,
        'add_to_cart': 0.10,
        'remove_from_cart': 0.03,
        'wishlist_add': 0.05,
        'wishlist_remove': 0.02,
        'filter_apply': 0.05,
        'sort_apply': 0.03,
    })
    
    # === CAMPAIGN TYPES ===
    campaign_types: Dict[str, float] = field(default_factory=lambda: {
        'DIWALI_SALE': 0.20,
        'SUMMER_SALE': 0.15,
        'NEW_YEAR': 0.12,
        'FLASH_SALE': 0.18,
        'CATEGORY_PROMO': 0.15,
        'LOYALTY_REWARD': 0.10,
        'WIN_BACK': 0.10,
    })
    
    # === BEHAVIORAL PARAMETERS ===
    # Average Order Value by segment
    aov_by_segment: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'High Value': {'mean': 3500, 'std': 1500},
        'Regular': {'mean': 1500, 'std': 800},
        'Occasional': {'mean': 1000, 'std': 500},
        'Window Shopper': {'mean': 800, 'std': 400},
        'Churned': {'mean': 1200, 'std': 600},
    })
    
    # Transactions per year by segment
    txn_per_year_by_segment: Dict[str, float] = field(default_factory=lambda: {
        'High Value': 18,
        'Regular': 10,
        'Occasional': 4,
        'Window Shopper': 1.5,
        'Churned': 2,
    })
    
    # Churn probability by segment
    churn_prob_by_segment: Dict[str, float] = field(default_factory=lambda: {
        'High Value': 0.05,
        'Regular': 0.15,
        'Occasional': 0.30,
        'Window Shopper': 0.50,
        'Churned': 0.85,
    })
    
    # Campaign click rate by segment
    campaign_ctr_by_segment: Dict[str, float] = field(default_factory=lambda: {
        'High Value': 0.25,
        'Regular': 0.15,
        'Occasional': 0.08,
        'Window Shopper': 0.03,
        'Churned': 0.02,
    })
    
    # Random seed
    seed: int = 42
    
    def save(self, path: str):
        """Save config to JSON."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                config_dict[key] = value
            else:
                config_dict[key] = value
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: str) -> 'DataGeneratorConfig':
        """Load config from JSON."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# ============================================================
# MAIN GENERATOR CLASS
# ============================================================

class SyntheticDataGenerator:
    """
    Generates synthetic data matching your exact table structure.
    
    Tables generated:
    1. demographics - User attributes (age, city, gender)
    2. transactions - Purchase history
    3. web_behavior - Web events (views, clicks, cart)
    4. campaigns - Campaign interactions (SPARSE)
    5. products - Product catalog (bonus)
    
    All tables are correlated through hidden user segments.
    """
    
    def __init__(self, config: Optional[DataGeneratorConfig] = None):
        """
        Initialize generator with configuration.
        
        Args:
            config: DataGeneratorConfig object. Uses defaults if None.
        """
        self.config = config or DataGeneratorConfig()
        self.rng = np.random.default_rng(self.config.seed)
        
        # Parse dates
        self.start_date = pd.to_datetime(self.config.start_date)
        self.end_date = pd.to_datetime(self.config.end_date)
        self.date_range_days = (self.end_date - self.start_date).days
        
        # Storage for generated data
        self.demographics = None
        self.transactions = None
        self.web_behavior = None
        self.campaigns = None
        self.products = None
        
        # Internal: user segment mapping (hidden, drives behavior)
        self._user_segments = None
        self._user_tenure = None
        
        print("="*60)
        print("Synthetic Data Generator Initialized")
        print("="*60)
        print(f"  Users: {self.config.num_users:,}")
        print(f"  Products: {self.config.num_products:,}")
        print(f"  Date Range: {self.start_date.date()} to {self.end_date.date()}")
        print(f"  Campaign Coverage: {self.config.campaign_coverage:.0%} (sparse)")
    
    # ================================================================
    # UTILITY METHODS
    # ================================================================
    
    def _sample_categorical(
        self, 
        distribution: Dict[str, float], 
        n: int
    ) -> np.ndarray:
        """Sample from categorical distribution."""
        categories = list(distribution.keys())
        probs = np.array(list(distribution.values()))
        probs = probs / probs.sum()  # Normalize
        return self.rng.choice(categories, size=n, p=probs)
    
    def _generate_timestamps(
        self,
        start: datetime,
        end: datetime,
        n: int,
        pattern: str = 'uniform'
    ) -> pd.DatetimeIndex:
        """
        Generate timestamps between start and end.
        
        Args:
            start: Start datetime
            end: End datetime
            n: Number of timestamps
            pattern: 'uniform', 'recent_bias', 'early_bias'
        """
        if start >= end:
            start = end - timedelta(days=30)
        
        total_seconds = (end - start).total_seconds()
        
        if pattern == 'uniform':
            random_seconds = self.rng.uniform(0, total_seconds, n)
        elif pattern == 'recent_bias':
            # More events recently (exponential)
            random_seconds = total_seconds * (1 - self.rng.exponential(0.3, n))
            random_seconds = np.clip(random_seconds, 0, total_seconds)
        elif pattern == 'early_bias':
            # More events early
            random_seconds = total_seconds * self.rng.exponential(0.3, n)
            random_seconds = np.clip(random_seconds, 0, total_seconds)
        else:
            random_seconds = self.rng.uniform(0, total_seconds, n)
        
        timestamps = [start + timedelta(seconds=s) for s in sorted(random_seconds)]
        return pd.DatetimeIndex(timestamps)
    
    # ================================================================
    # STEP 1: GENERATE DEMOGRAPHICS TABLE
    # ================================================================
    
    def generate_demographics(self) -> pd.DataFrame:
        """
        Generate demographics table.
        
        Columns:
        - user_id: Unique user identifier
        - age: User age (18-70)
        - age_group: Bucketed age
        - city: User city
        - gender: User gender
        - registration_date: When user registered
        
        Also generates hidden segment assignments.
        """
        print("\n[1/5] Generating Demographics Table...")
        
        n = self.config.num_users
        
        # Generate user IDs
        user_ids = [f"U{i:07d}" for i in range(n)]
        
        # Assign hidden segments (drives all other behavior)
        self._user_segments = self._sample_categorical(
            self.config.user_segments, n
        )
        
        # Generate demographics
        cities = self._sample_categorical(self.config.cities, n)
        age_groups = self._sample_categorical(self.config.age_groups, n)
        genders = self._sample_categorical(self.config.genders, n)
        
        # Convert age groups to actual ages
        age_ranges = {
            '18-24': (18, 24),
            '25-34': (25, 34),
            '35-44': (35, 44),
            '45-54': (45, 54),
            '55+': (55, 70),
        }
        ages = np.array([
            self.rng.integers(age_ranges[ag][0], age_ranges[ag][1] + 1)
            for ag in age_groups
        ])
        
        # Generate registration dates (tenure)
        # High value users tend to have longer tenure
        tenure_days = np.zeros(n)
        for segment in self.config.user_segments.keys():
            mask = self._user_segments == segment
            
            if segment == 'High Value':
                tenure_days[mask] = self.rng.exponential(600, mask.sum())
            elif segment == 'Regular':
                tenure_days[mask] = self.rng.exponential(400, mask.sum())
            elif segment == 'Occasional':
                tenure_days[mask] = self.rng.exponential(300, mask.sum())
            elif segment == 'Window Shopper':
                tenure_days[mask] = self.rng.exponential(200, mask.sum())
            elif segment == 'Churned':
                tenure_days[mask] = self.rng.exponential(500, mask.sum())
        
        tenure_days = np.clip(tenure_days, 7, self.date_range_days)
        self._user_tenure = tenure_days
        
        registration_dates = self.end_date - pd.to_timedelta(tenure_days, unit='D')
        
        # Create DataFrame
        self.demographics = pd.DataFrame({
            'user_id': user_ids,
            'age': ages,
            'age_group': age_groups,
            'city': cities,
            'gender': genders,
            'registration_date': registration_dates,
        })
        
        print(f"  ✓ Generated {len(self.demographics):,} user records")
        print(f"  ✓ Columns: {list(self.demographics.columns)}")
        
        return self.demographics
    
    # ================================================================
    # STEP 2: GENERATE PRODUCTS TABLE
    # ================================================================
    
    def generate_products(self) -> pd.DataFrame:
        """
        Generate products table.
        
        Columns:
        - product_id: Unique product identifier
        - category: Product category
        - price: Product price
        - created_date: When product was added
        """
        print("\n[2/5] Generating Products Table...")
        
        n = self.config.num_products
        
        # Generate product IDs
        product_ids = [f"P{i:06d}" for i in range(n)]
        
        # Categories
        categories = self._sample_categorical(self.config.product_categories, n)
        
        # Prices by category
        category_price_ranges = {
            'Electronics': (500, 50000),
            'Fashion': (200, 5000),
            'Home & Living': (300, 20000),
            'Beauty': (100, 3000),
            'Sports': (200, 10000),
            'Books': (100, 2000),
            'Food & Grocery': (50, 1000),
            'Others': (100, 5000),
        }
        
        prices = np.array([
            self.rng.uniform(*category_price_ranges.get(cat, (100, 5000)))
            for cat in categories
        ])
        
        # Created dates
        created_dates = self._generate_timestamps(
            self.start_date,
            self.end_date - timedelta(days=30),
            n,
            pattern='early_bias'
        )
        
        self.products = pd.DataFrame({
            'product_id': product_ids,
            'category': categories,
            'price': np.round(prices, 2),
            'created_date': created_dates,
        })
        
        print(f"  ✓ Generated {len(self.products):,} products")
        print(f"  ✓ Categories: {list(self.config.product_categories.keys())}")
        
        return self.products
    
    # ================================================================
    # STEP 3: GENERATE TRANSACTIONS TABLE
    # ================================================================
    
    def generate_transactions(self) -> pd.DataFrame:
        """
        Generate transactions table.
        
        Columns:
        - transaction_id: Unique transaction identifier
        - user_id: User who made the purchase
        - product_id: Product purchased
        - timestamp: When the transaction occurred
        - amount: Transaction amount
        - quantity: Number of items (1-5)
        
        Transaction volume and amount vary by user segment.
        """
        print("\n[3/5] Generating Transactions Table...")
        
        if self.demographics is None:
            raise ValueError("Generate demographics first!")
        if self.products is None:
            raise ValueError("Generate products first!")
        
        transactions_list = []
        
        user_ids = self.demographics['user_id'].values
        product_ids = self.products['product_id'].values
        product_prices = self.products['price'].values
        
        for i, (user_id, segment, tenure) in enumerate(zip(
            user_ids, self._user_segments, self._user_tenure
        )):
            if i % 20000 == 0:
                print(f"  Processing user {i:,}/{len(user_ids):,}...")
            
            # Number of transactions based on segment
            txn_per_year = self.config.txn_per_year_by_segment[segment]
            years = tenure / 365
            expected_txns = txn_per_year * years
            
            # Add variance
            num_txns = max(0, int(self.rng.poisson(expected_txns)))
            
            if num_txns == 0:
                continue
            
            # Generate transactions for this user
            registration_date = self.demographics.loc[i, 'registration_date']
            
            # Timestamps
            timestamps = self._generate_timestamps(
                registration_date,
                self.end_date,
                num_txns,
                pattern='recent_bias' if segment != 'Churned' else 'early_bias'
            )
            
            # Products (some users have category preferences)
            preferred_category = self.rng.choice(list(self.config.product_categories.keys()))
            preferred_mask = self.products['category'] == preferred_category
            preferred_products = product_ids[preferred_mask]
            preferred_prices = product_prices[preferred_mask]
            
            for ts in timestamps:
                # 60% chance to buy from preferred category
                if len(preferred_products) > 0 and self.rng.random() < 0.6:
                    idx = self.rng.integers(0, len(preferred_products))
                    product_id = preferred_products[idx]
                    base_price = preferred_prices[idx]
                else:
                    idx = self.rng.integers(0, len(product_ids))
                    product_id = product_ids[idx]
                    base_price = product_prices[idx]
                
                # Amount based on segment
                aov_params = self.config.aov_by_segment[segment]
                amount_multiplier = self.rng.normal(1.0, 0.3)
                amount = base_price * max(0.5, amount_multiplier)
                
                # Quantity
                quantity = self.rng.choice([1, 1, 1, 2, 2, 3], p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
                
                transactions_list.append({
                    'user_id': user_id,
                    'product_id': product_id,
                    'timestamp': ts,
                    'amount': round(amount * quantity, 2),
                    'quantity': quantity,
                })
        
        # Create DataFrame
        self.transactions = pd.DataFrame(transactions_list)
        
        # Add transaction IDs
        self.transactions['transaction_id'] = [
            f"TXN{i:09d}" for i in range(len(self.transactions))
        ]
        
        # Reorder columns
        self.transactions = self.transactions[[
            'transaction_id', 'user_id', 'product_id', 
            'timestamp', 'amount', 'quantity'
        ]]
        
        # Sort by timestamp
        self.transactions = self.transactions.sort_values('timestamp').reset_index(drop=True)
        
        print(f"  ✓ Generated {len(self.transactions):,} transactions")
        print(f"  ✓ Users with transactions: {self.transactions['user_id'].nunique():,}")
        print(f"  ✓ Avg transactions/user: {len(self.transactions)/self.config.num_users:.1f}")
        
        return self.transactions
    
    # ================================================================
    # STEP 4: GENERATE WEB BEHAVIOR TABLE
    # ================================================================
    
    def generate_web_behavior(self) -> pd.DataFrame:
        """
        Generate web behavior table.
        
        Columns:
        - event_id: Unique event identifier
        - user_id: User who performed the action
        - product_id: Product involved (if applicable)
        - event_type: Type of event (view, cart, etc.)
        - timestamp: When the event occurred
        - session_id: Session identifier
        
        Web activity varies by user segment.
        """
        print("\n[4/5] Generating Web Behavior Table...")
        
        if self.demographics is None:
            raise ValueError("Generate demographics first!")
        if self.products is None:
            raise ValueError("Generate products first!")
        
        events_list = []
        
        user_ids = self.demographics['user_id'].values
        product_ids = self.products['product_id'].values
        
        # Web activity multipliers by segment
        web_multipliers = {
            'High Value': 1.5,
            'Regular': 1.2,
            'Occasional': 1.0,
            'Window Shopper': 2.0,  # Browse a lot but don't buy
            'Churned': 0.3,
        }
        
        for i, (user_id, segment, tenure) in enumerate(zip(
            user_ids, self._user_segments, self._user_tenure
        )):
            if i % 20000 == 0:
                print(f"  Processing user {i:,}/{len(user_ids):,}...")
            
            # Number of web events based on segment
            multiplier = web_multipliers[segment]
            years = tenure / 365
            expected_events = self.config.avg_web_events_per_user * multiplier * years
            
            num_events = max(0, int(self.rng.poisson(expected_events)))
            
            if num_events == 0:
                continue
            
            # Cap events
            num_events = min(num_events, 500)
            
            # Generate events
            registration_date = self.demographics.loc[i, 'registration_date']
            
            timestamps = self._generate_timestamps(
                registration_date,
                self.end_date,
                num_events,
                pattern='recent_bias' if segment != 'Churned' else 'early_bias'
            )
            
            # Assign sessions (new session if > 30 min gap)
            session_id = 0
            current_session = f"{user_id}_S{session_id:04d}"
            last_ts = timestamps[0]
            
            # Preferred products for this user
            preferred_category = self.rng.choice(list(self.config.product_categories.keys()))
            preferred_mask = self.products['category'] == preferred_category
            preferred_products = product_ids[preferred_mask]
            
            for ts in timestamps:
                # Check for new session
                if (ts - last_ts).total_seconds() > 1800:  # 30 min
                    session_id += 1
                    current_session = f"{user_id}_S{session_id:04d}"
                
                last_ts = ts
                
                # Event type
                event_type = self._sample_categorical(
                    self.config.web_event_types, 1
                )[0]
                
                # Product (70% preferred category)
                if len(preferred_products) > 0 and self.rng.random() < 0.7:
                    product_id = self.rng.choice(preferred_products)
                else:
                    product_id = self.rng.choice(product_ids)
                
                events_list.append({
                    'user_id': user_id,
                    'product_id': product_id,
                    'event_type': event_type,
                    'timestamp': ts,
                    'session_id': current_session,
                })
        
        # Create DataFrame
        self.web_behavior = pd.DataFrame(events_list)
        
        # Add event IDs
        self.web_behavior['event_id'] = [
            f"EVT{i:010d}" for i in range(len(self.web_behavior))
        ]
        
        # Reorder columns
        self.web_behavior = self.web_behavior[[
            'event_id', 'user_id', 'product_id', 
            'event_type', 'timestamp', 'session_id'
        ]]
        
        # Sort by timestamp
        self.web_behavior = self.web_behavior.sort_values('timestamp').reset_index(drop=True)
        
        print(f"  ✓ Generated {len(self.web_behavior):,} web events")
        print(f"  ✓ Users with web activity: {self.web_behavior['user_id'].nunique():,}")
        print(f"  ✓ Event types: {self.web_behavior['event_type'].nunique()}")
        
        return self.web_behavior
    
    # ================================================================
    # STEP 5: GENERATE CAMPAIGNS TABLE (SPARSE)
    # ================================================================
    
    def generate_campaigns(self) -> pd.DataFrame:
        """
        Generate campaigns table (SPARSE).
        
        Columns:
        - campaign_id: Campaign identifier
        - user_id: User who received the campaign
        - timestamp: When campaign was sent
        - clicked: Whether user clicked (0/1)
        
        Only a fraction of users have campaign data (sparse by design).
        """
        print("\n[5/5] Generating Campaigns Table (Sparse)...")
        
        if self.demographics is None:
            raise ValueError("Generate demographics first!")
        
        campaigns_list = []
        
        # Select users who will have campaign data
        num_exposed_users = int(self.config.num_users * self.config.campaign_coverage)
        
        # Bias toward more active users
        # High Value and Regular users more likely to be in campaigns
        segment_campaign_prob = {
            'High Value': 0.5,
            'Regular': 0.3,
            'Occasional': 0.15,
            'Window Shopper': 0.08,
            'Churned': 0.05,
        }
        
        campaign_probs = np.array([
            segment_campaign_prob[seg] for seg in self._user_segments
        ])
        campaign_probs = campaign_probs / campaign_probs.sum()
        
        exposed_user_indices = self.rng.choice(
            len(self.demographics),
            size=num_exposed_users,
            replace=False,
            p=campaign_probs
        )
        
        exposed_users = self.demographics.iloc[exposed_user_indices]
        
        print(f"  Campaign exposed users: {len(exposed_users):,} ({self.config.campaign_coverage:.0%})")
        
        for idx in exposed_user_indices:
            user_id = self.demographics.loc[idx, 'user_id']
            segment = self._user_segments[idx]
            registration_date = self.demographics.loc[idx, 'registration_date']
            
            # Number of campaigns this user received
            num_campaigns = max(1, int(self.rng.poisson(
                self.config.avg_campaigns_per_exposed_user
            )))
            num_campaigns = min(num_campaigns, 10)  # Cap at 10
            
            # Campaign timestamps (after registration)
            if registration_date >= self.end_date:
                continue
            
            timestamps = self._generate_timestamps(
                registration_date + timedelta(days=7),
                self.end_date,
                num_campaigns,
                pattern='uniform'
            )
            
            # Click probability based on segment
            click_prob = self.config.campaign_ctr_by_segment[segment]
            
            for ts in timestamps:
                campaign_type = self._sample_categorical(
                    self.config.campaign_types, 1
                )[0]
                
                clicked = 1 if self.rng.random() < click_prob else 0
                
                campaigns_list.append({
                    'user_id': user_id,
                    'campaign_id': campaign_type,
                    'timestamp': ts,
                    'clicked': clicked,
                })
        
        # Create DataFrame
        self.campaigns = pd.DataFrame(campaigns_list)
        
        # Sort by timestamp
        self.campaigns = self.campaigns.sort_values('timestamp').reset_index(drop=True)
        
        print(f"  ✓ Generated {len(self.campaigns):,} campaign records")
        print(f"  ✓ Users with campaigns: {self.campaigns['user_id'].nunique():,}")
        print(f"  ✓ Overall CTR: {self.campaigns['clicked'].mean():.1%}")
        
        return self.campaigns
    
    # ================================================================
    # MAIN GENERATION METHOD
    # ================================================================
    
    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """
        Generate all tables.
        
        Returns:
            Dictionary with all tables:
            - demographics
            - products
            - transactions
            - web_behavior
            - campaigns
        """
        print("\n" + "="*60)
        print("GENERATING ALL SYNTHETIC DATA TABLES")
        print("="*60)
        
        self.generate_demographics()
        self.generate_products()
        self.generate_transactions()
        self.generate_web_behavior()
        self.generate_campaigns()
        
        print("\n" + "="*60)
        print("GENERATION COMPLETE")
        print("="*60)
        
        return {
            'demographics': self.demographics,
            'products': self.products,
            'transactions': self.transactions,
            'web_behavior': self.web_behavior,
            'campaigns': self.campaigns,
        }
    
    # ================================================================
    # STATISTICS & VALIDATION
    # ================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated data."""
        stats = {}
        
        if self.demographics is not None:
            stats['demographics'] = {
                'num_users': len(self.demographics),
                'cities': self.demographics['city'].value_counts().to_dict(),
                'age_groups': self.demographics['age_group'].value_counts().to_dict(),
                'genders': self.demographics['gender'].value_counts().to_dict(),
            }
        
        if self.products is not None:
            stats['products'] = {
                'num_products': len(self.products),
                'categories': self.products['category'].value_counts().to_dict(),
                'avg_price': self.products['price'].mean(),
            }
        
        if self.transactions is not None:
            stats['transactions'] = {
                'num_transactions': len(self.transactions),
                'unique_users': self.transactions['user_id'].nunique(),
                'unique_products': self.transactions['product_id'].nunique(),
                'avg_amount': self.transactions['amount'].mean(),
                'total_revenue': self.transactions['amount'].sum(),
                'avg_txn_per_user': len(self.transactions) / self.config.num_users,
            }
        
        if self.web_behavior is not None:
            stats['web_behavior'] = {
                'num_events': len(self.web_behavior),
                'unique_users': self.web_behavior['user_id'].nunique(),
                'event_types': self.web_behavior['event_type'].value_counts().to_dict(),
                'avg_events_per_user': len(self.web_behavior) / self.config.num_users,
            }
        
        if self.campaigns is not None:
            stats['campaigns'] = {
                'num_records': len(self.campaigns),
                'unique_users': self.campaigns['user_id'].nunique(),
                'coverage': self.campaigns['user_id'].nunique() / self.config.num_users,
                'overall_ctr': self.campaigns['clicked'].mean(),
                'campaign_types': self.campaigns['campaign_id'].value_counts().to_dict(),
            }
        
        return stats
    
    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("DATA STATISTICS")
        print("="*60)
        
        if 'demographics' in stats:
            d = stats['demographics']
            print(f"\n📊 DEMOGRAPHICS")
            print(f"   Users: {d['num_users']:,}")
        
        if 'products' in stats:
            p = stats['products']
            print(f"\n📦 PRODUCTS")
            print(f"   Products: {p['num_products']:,}")
            print(f"   Avg Price: ₹{p['avg_price']:,.2f}")
        
        if 'transactions' in stats:
            t = stats['transactions']
            print(f"\n💳 TRANSACTIONS")
            print(f"   Total: {t['num_transactions']:,}")
            print(f"   Users with purchases: {t['unique_users']:,}")
            print(f"   Avg per user: {t['avg_txn_per_user']:.1f}")
            print(f"   Avg amount: ₹{t['avg_amount']:,.2f}")
            print(f"   Total revenue: ₹{t['total_revenue']:,.2f}")
        
        if 'web_behavior' in stats:
            w = stats['web_behavior']
            print(f"\n🌐 WEB BEHAVIOR")
            print(f"   Total events: {w['num_events']:,}")
            print(f"   Users with activity: {w['unique_users']:,}")
            print(f"   Avg events/user: {w['avg_events_per_user']:.1f}")
        
        if 'campaigns' in stats:
            c = stats['campaigns']
            print(f"\n📧 CAMPAIGNS (Sparse)")
            print(f"   Total records: {c['num_records']:,}")
            print(f"   Users exposed: {c['unique_users']:,} ({c['coverage']:.1%})")
            print(f"   Overall CTR: {c['overall_ctr']:.1%}")
    
    # ================================================================
    # SAVE DATA
    # ================================================================
    
    def save_data(
        self, 
        output_dir: str, 
        format: str = 'csv',
        compression: Optional[str] = None
    ):
        """
        Save all tables to files.
        
        Args:
            output_dir: Output directory path
            format: 'csv' or 'parquet'
            compression: Optional compression ('gzip' for csv, 'snappy' for parquet)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        tables = {
            'demographics': self.demographics,
            'products': self.products,
            'transactions': self.transactions,
            'web_behavior': self.web_behavior,
            'campaigns': self.campaigns,
        }
        
        print(f"\nSaving data to {output_dir}/")
        
        for name, df in tables.items():
            if df is not None:
                if format == 'csv':
                    path = f"{output_dir}/{name}.csv"
                    if compression:
                        path += '.gz'
                        df.to_csv(path, index=False, compression='gzip')
                    else:
                        df.to_csv(path, index=False)
                elif format == 'parquet':
                    path = f"{output_dir}/{name}.parquet"
                    df.to_parquet(path, index=False, compression=compression or 'snappy')
                
                print(f"  ✓ Saved {name} ({len(df):,} rows)")
        
        # Save config
        self.config.save(f"{output_dir}/config.json")
        print(f"  ✓ Saved config.json")
    
    # ================================================================
    # CALCULATE TARGETS (for validation)
    # ================================================================
    
    def calculate_ground_truth_targets(
        self,
        churn_threshold_days: int = 90
    ) -> pd.DataFrame:
        """
        Calculate ground truth targets for validation.
        
        Args:
            churn_threshold_days: Days without purchase to be considered churned
        
        Returns:
            DataFrame with user_id and target variables
        """
        if self.transactions is None:
            raise ValueError("Generate transactions first!")
        
        print("\nCalculating ground truth targets...")
        
        targets = pd.DataFrame({
            'user_id': self.demographics['user_id']
        })
        
        # Get last purchase per user
        last_purchase = self.transactions.groupby('user_id')['timestamp'].max()
        targets['last_purchase'] = targets['user_id'].map(last_purchase)
        
        # Churn
        targets['recency_days'] = (self.end_date - targets['last_purchase']).dt.days
        targets['churned'] = (targets['recency_days'] > churn_threshold_days).astype(int)
        targets['churned'] = targets['churned'].fillna(1)  # No purchase = churned
        
        # CLV (last 365 days)
        cutoff = self.end_date - timedelta(days=365)
        clv = self.transactions[
            self.transactions['timestamp'] >= cutoff
        ].groupby('user_id')['amount'].sum()
        targets['clv'] = targets['user_id'].map(clv).fillna(0)
        
        # Total spend
        total_spend = self.transactions.groupby('user_id')['amount'].sum()
        targets['total_spend'] = targets['user_id'].map(total_spend).fillna(0)
        
        # Order count
        order_count = self.transactions.groupby('user_id').size()
        targets['order_count'] = targets['user_id'].map(order_count).fillna(0)
        
        # Add segment (for analysis)
        targets['segment'] = self._user_segments
        
        # Drop intermediate columns
        targets = targets.drop(columns=['last_purchase', 'recency_days'])
        
        print(f"  ✓ Churn rate: {targets['churned'].mean():.1%}")
        print(f"  ✓ Avg CLV: ₹{targets['clv'].mean():,.2f}")
        
        return targets


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    
    # Create configuration
    config = DataGeneratorConfig(
        num_users=10_000,       # Start small for testing
        num_products=1_000,
        campaign_coverage=0.15,  # 15% sparse campaign data
        seed=42,
    )
    
    # Initialize generator
    generator = SyntheticDataGenerator(config)
    
    # Generate all tables
    data = generator.generate_all()
    
    # Print statistics
    generator.print_statistics()
    
    # Calculate ground truth targets
    targets = generator.calculate_ground_truth_targets()
    
    print("\n" + "="*60)
    print("TARGET STATISTICS BY SEGMENT")
    print("="*60)
    segment_stats = targets.groupby('segment').agg({
        'churned': 'mean',
        'clv': 'mean',
        'total_spend': 'mean',
        'order_count': 'mean',
    }).round(2)
    print(segment_stats.to_string())
    
    # Preview each table
    print("\n" + "="*60)
    print("TABLE PREVIEWS")
    print("="*60)
    
    for name, df in data.items():
        print(f"\n📋 {name.upper()} (first 3 rows):")
        print(df.head(3).to_string())
    
    # Save data
    generator.save_data('./synthetic_data', format='csv')
