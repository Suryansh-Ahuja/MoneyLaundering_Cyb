import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import os

class AdvancedFanDetector:
    """
    Advanced FAN-OUT and FAN-IN Money Laundering Pattern Detector
    Multi-dimensional analysis for both distribution and consolidation patterns
    """

    def __init__(self, csv_file='env_name/LI-Small_Trans.csv'):
        print("🎯 ADVANCED FAN-OUT & FAN-IN MONEY LAUNDERING DETECTOR")
        print("="*65)
        print("🔍 Detecting multi-currency, multi-degree fan patterns")
        print("📤 FAN-OUT: One sender → Many receivers (Distribution)")
        print("📥 FAN-IN: Many senders → One receiver (Consolidation)")

        try:
            self.df = pd.read_csv(csv_file)
            print(f"✅ Loaded {len(self.df):,} transactions")
        except:
            print("❌ Could not load CSV file")
            self.df = pd.DataFrame()
            return

        # Prepare data
        self._prepare_data()
        self.detected_fanout_patterns = []
        self.detected_fanin_patterns = []

        # Risk thresholds for both patterns
        self.risk_matrix = {
            'fan_thresholds': {3: 20, 7: 40, 12: 60, 16: 80},
            'currency_thresholds': {1: 0, 2: 15, 3: 25, 4: 35, 5: 45},
            'amount_thresholds': {5000: 10, 20000: 20, 50000: 30, 200000: 40},
            'time_thresholds': {7: 25, 14: 20, 21: 15, 30: 10}
        }

    def _prepare_data(self):
        """Prepare data with proper handling of the specific column structure"""

        print("🔧 Mapping dataset columns...")

        # Map specific columns to standard names - Adjusted for HI-Small_Trans.csv
        column_mapping = {
            'Account': 'Account',
            'Account.1': 'Account.1',
            'Amount Received': 'Amount',
            'Receiving Currency': 'Currency',
            'Timestamp': 'Timestamp'
        }

        # Apply column mapping
        temp_df = pd.DataFrame()
        for old_col, new_col in column_mapping.items():
            if old_col in self.df.columns:
                temp_df[new_col] = self.df[old_col]
                print(f"   ✅ Mapped {old_col} → {new_col}")
            else:
                print(f"   ⚠️ Column {old_col} not found")

        self.df = temp_df

        # Handle timestamps
        if 'Timestamp' in self.df.columns:
            try:
                self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], errors='coerce')
                self.df.dropna(subset=['Timestamp'], inplace=True)
                print("   ✅ Timestamp processed and invalid rows removed")
            except Exception as e:
                print(f"   ❌ Error processing existing Timestamp: {e}")
                if 'Timestamp' in self.df.columns:
                    self.df.drop(columns=['Timestamp'], inplace=True)
                print("   ⚠️ Proceeding without Timestamp for temporal analysis.")
        else:
            print("   ⚠️ Timestamp column not found. Proceeding without temporal analysis.")

        # Handle amounts

        currency_rate = {
            'US Dollar' : 1.0, 
            'Rupee' : 0.012,
            'Yuan' : 0.14,
            'Euro' : 1.3,
            'Bitcoin' : 108547.30,
        }

        if 'Amount' in self.df.columns:
            self.df['Amount'] = pd.to_numeric(self.df['Amount'], errors='coerce').fillna(0)
            print("   ✅ Amount column processed")

            self.df['Amount'] = self.df['Amount'] * self.df['Currency'].map(currency_rate).fillna(1)
        else:
            print("   ❌ Amount column not found after mapping.")
            self.df = pd.DataFrame()
            return

        # Handle currency
        if 'Currency' not in self.df.columns:
            self.df['Currency'] = 'USD'
            print("   ⚠️ Currency column not found after mapping. Using default USD.")


        
        

        print(f"\n📊 Data prepared: {len(self.df):,} transactions")

        # Verify required columns exist
        required_cols = ['Account', 'Account.1', 'Amount', 'Currency']
        missing_cols = [col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            print(f"❌ Missing required columns after preparation: {missing_cols}")
            self.df = pd.DataFrame()
            return

        if 'Timestamp' in self.df.columns and not self.df['Timestamp'].empty:
            print(f"⏰ Time range: {self.df['Timestamp'].min()} to {self.df['Timestamp'].max()}")
        else:
            print("⏰ Timestamp column not available. Temporal analysis will be skipped.")

        # Show data summary
        print(f"\n📋 DATA SUMMARY:")
        if not self.df.empty:
            print(f"   • Unique senders: {self.df['Account'].nunique():,}")
            print(f"   • Unique receivers: {self.df['Account.1'].nunique():,}")
            print(f"   • Total amount: ${self.df['Amount'].sum():,.2f}")

            # Currency distribution
            if 'Currency' in self.df.columns:
                currency_dist = self.df['Currency'].value_counts()
                print(f"   • Currencies found: {list(currency_dist.index[:5])}")
        else:
            print("   DataFrame is empty after preparation.")

    def detect_all_fan_patterns(self,
                               min_connections=3,
                               max_time_window_days=30,
                               min_total_amount=10000,
                               max_accounts_to_analyze=100):
        """
        Detect both FAN-OUT and FAN-IN patterns
        """
        print(f"\n🔍 DETECTING BOTH FAN-OUT & FAN-IN PATTERNS")
        print("="*55)
        print(f"Parameters:")
        print(f"  • Min connections: {min_connections}")
        print(f"  • Max time window: {max_time_window_days} days")
        print(f"  • Min total amount: ${min_total_amount:,}")

        if self.df.empty:
            print("❌ Cannot proceed due to missing data.")
            return [], []

        # Detect FAN-OUT patterns (one sender → many receivers)
        print(f"\n📤 DETECTING FAN-OUT PATTERNS...")
        fanout_patterns = self.detect_fanout_patterns(
            min_connections, max_time_window_days, min_total_amount, max_accounts_to_analyze
        )

        # Detect FAN-IN patterns (many senders → one receiver)
        print(f"\n📥 DETECTING FAN-IN PATTERNS...")
        fanin_patterns = self.detect_fanin_patterns(
            min_connections, max_time_window_days, min_total_amount, max_accounts_to_analyze
        )

        self.detected_fanout_patterns = fanout_patterns
        self.detected_fanin_patterns = fanin_patterns

        print(f"\n✅ DETECTION COMPLETE")
        print(f"📤 Found {len(fanout_patterns)} FAN-OUT patterns")
        print(f"📥 Found {len(fanin_patterns)} FAN-IN patterns")
        print(f"🎯 Total suspicious patterns: {len(fanout_patterns) + len(fanin_patterns)}")

        return fanout_patterns, fanin_patterns

    def detect_fanout_patterns(self, min_recipients, max_time_window, min_total_amount, max_accounts):
        """Detect FAN-OUT patterns (one sender → many receivers)"""

        patterns = []

        # Get sender statistics
        sender_stats = self._get_sender_statistics()
        if len(sender_stats) == 0:
            return []

        # Filter potential fan-out accounts
        potential_accounts = sender_stats[
            (sender_stats['unique_recipients'] >= min_recipients) &
            (sender_stats['total_amount'] >= min_total_amount)&
            (sender_stats['total_amount'] <= 10000000)
        ].sort_values(['unique_recipients', 'total_amount'], ascending=False)

        print(f"   📤 Found {len(potential_accounts)} potential FAN-OUT accounts")

        # Analyze each account
        for i, (_, account_info) in enumerate(potential_accounts.head(max_accounts).iterrows()):
            account = account_info['Account']

            try:
                pattern = self._analyze_fanout_pattern(account, account_info, max_time_window)
                if pattern:
                    patterns.append(pattern)
                    #print(f"    🚨 FAN-OUT: {account} (Risk: {pattern['risk_score']:.1f})")
            except Exception as e:
                continue

        patterns.sort(key=lambda x: x['risk_score'], reverse=True)
        return patterns

    def detect_fanin_patterns(self, min_senders, max_time_window, min_total_amount, max_accounts):
        """Detect FAN-IN patterns (many senders → one receiver)"""

        patterns = []

        # Get receiver statistics
        receiver_stats = self._get_receiver_statistics()
        if len(receiver_stats) == 0:
            return []

        # Filter potential fan-in accounts
        potential_accounts = receiver_stats[
            (receiver_stats['unique_senders'] >= min_senders) &
            (receiver_stats['total_amount'] >= min_total_amount)&
            (receiver_stats['total_amount'] <= 10000000)
        ].sort_values(['unique_senders', 'total_amount'], ascending=False)

        print(f"   📥 Found {len(potential_accounts)} potential FAN-IN accounts")

        # Analyze each account
        for i, (_, account_info) in enumerate(potential_accounts.head(max_accounts).iterrows()):
            account = account_info['Account.1']

            try:
                pattern = self._analyze_fanin_pattern(account, account_info, max_time_window)
                if pattern:
                    patterns.append(pattern)
                    #print(f"    🚨 FAN-IN: {account} (Risk: {pattern['risk_score']:.1f})")
            except Exception as e:
                continue

        patterns.sort(key=lambda x: x['risk_score'], reverse=True)
        return patterns

    def _get_sender_statistics(self):
        """Get sender statistics for FAN-OUT detection"""

        try:
            agg_dict = {
                'Account.1': 'nunique',
                'Amount': ['sum', 'count', 'mean', 'std'],
                'Currency': lambda x: x.nunique()
            }

            if 'Timestamp' in self.df.columns and not self.df['Timestamp'].empty:
                agg_dict['Timestamp'] = ['min', 'max']

            sender_stats = self.df.groupby('Account').agg(agg_dict).reset_index()

            # Flatten column names
            if 'Timestamp' in self.df.columns and 'Timestamp' in sender_stats.columns.get_level_values(0):
                sender_stats.columns = ['_'.join(col).strip('_') for col in sender_stats.columns.values]
                sender_stats.rename(columns={
                    'Account_': 'Account',
                    'Account.1_nunique': 'unique_recipients',
                    'Amount_sum': 'total_amount',
                    'Amount_count': 'txn_count',
                    'Amount_mean': 'mean_amount',
                    'Amount_std': 'std_amount',
                    'Currency_<lambda>': 'unique_currencies',
                    'Timestamp_min': 'first_txn',
                    'Timestamp_max': 'last_txn'
                }, inplace=True)

                sender_stats['time_span_days'] = (
                    sender_stats['last_txn'] - sender_stats['first_txn']
                ).dt.days.fillna(0)
            else:
                sender_stats.columns = ['Account', 'unique_recipients', 'total_amount', 'txn_count',
                                      'mean_amount', 'std_amount', 'unique_currencies']
                sender_stats['time_span_days'] = 0

            return sender_stats

        except Exception as e:
            print(f"    ❌ Error generating sender statistics: {e}")
            return pd.DataFrame()

    def _get_receiver_statistics(self):
        """Get receiver statistics for FAN-IN detection"""

        try:
            agg_dict = {
                'Account': 'nunique',
                'Amount': ['sum', 'count', 'mean', 'std'],
                'Currency': lambda x: x.nunique()
            }

            if 'Timestamp' in self.df.columns and not self.df['Timestamp'].empty:
                agg_dict['Timestamp'] = ['min', 'max']

            receiver_stats = self.df.groupby('Account.1').agg(agg_dict).reset_index()

            # Flatten column names
            if 'Timestamp' in self.df.columns and 'Timestamp' in receiver_stats.columns.get_level_values(0):
                receiver_stats.columns = ['_'.join(col).strip('_') for col in receiver_stats.columns.values]
                receiver_stats.rename(columns={
                    'Account.1_': 'Account.1',
                    'Account_nunique': 'unique_senders',
                    'Amount_sum': 'total_amount',
                    'Amount_count': 'txn_count',
                    'Amount_mean': 'mean_amount',
                    'Amount_std': 'std_amount',
                    'Currency_<lambda>': 'unique_currencies',
                    'Timestamp_min': 'first_txn',
                    'Timestamp_max': 'last_txn'
                }, inplace=True)

                receiver_stats['time_span_days'] = (
                    receiver_stats['last_txn'] - receiver_stats['first_txn']
                ).dt.days.fillna(0)
            else:
                receiver_stats.columns = ['Account.1', 'unique_senders', 'total_amount', 'txn_count',
                                        'mean_amount', 'std_amount', 'unique_currencies']
                receiver_stats['time_span_days'] = 0

            return receiver_stats

        except Exception as e:
            print(f"    ❌ Error generating receiver statistics: {e}")
            return pd.DataFrame()

    def _analyze_fanout_pattern(self, account, account_stats, max_time_window):
        """Analyze individual FAN-OUT pattern"""

        # Get all outgoing transactions from this account
        account_txns = self.df[self.df['Account'] == account].copy()

        if 'Timestamp' in self.df.columns and not self.df['Timestamp'].empty:
            account_txns = account_txns.sort_values('Timestamp')

        if len(account_txns) < 3:
            return None

        # Time window check
        if 'time_span_days' in account_stats and account_stats['time_span_days'] > max_time_window:
            return None

        # Pattern analysis
        pattern_analysis = {
            'pattern_type': 'FAN-OUT',
            'account': account,
            'fan_degree': account_stats['unique_recipients'],
            'transaction_count': account_stats['txn_count'],
            'total_amount': account_stats['total_amount'],
            'time_span_days': account_stats.get('time_span_days', 0),
            'unique_currencies': account_stats['unique_currencies'],
            'transactions': account_txns,
            'connection_type': 'recipients'
        }

        # Add analysis components
        currency_analysis = self._analyze_currency_patterns(account_txns)
        pattern_analysis.update(currency_analysis)

        amount_analysis = self._analyze_amount_patterns(account_txns)
        pattern_analysis.update(amount_analysis)

        if 'Timestamp' in self.df.columns and not self.df['Timestamp'].empty:
            temporal_analysis = self._analyze_temporal_patterns(account_txns)
            pattern_analysis.update(temporal_analysis)
        else:
            pattern_analysis.update({'temporal_clustering_score': 0, 'transaction_velocity': 0})

        # Calculate risk score
        risk_score = self._calculate_fan_risk_score(pattern_analysis)
        pattern_analysis['risk_score'] = risk_score

        if risk_score >= 50:
            return pattern_analysis

        return None

    def _analyze_fanin_pattern(self, account, account_stats, max_time_window):
        """Analyze individual FAN-IN pattern"""

        # Get all incoming transactions to this account
        account_txns = self.df[self.df['Account.1'] == account].copy()

        if 'Timestamp' in self.df.columns and not self.df['Timestamp'].empty:
            account_txns = account_txns.sort_values('Timestamp')

        if len(account_txns) < 3:
            return None

        # Time window check
        if 'time_span_days' in account_stats and account_stats['time_span_days'] > max_time_window:
            return None

        # Pattern analysis
        pattern_analysis = {
            'pattern_type': 'FAN-IN',
            'account': account,
            'fan_degree': account_stats['unique_senders'],
            'transaction_count': account_stats['txn_count'],
            'total_amount': account_stats['total_amount'],
            'time_span_days': account_stats.get('time_span_days', 0),
            'unique_currencies': account_stats['unique_currencies'],
            'transactions': account_txns,
            'connection_type': 'senders'
        }

        # Add analysis components
        currency_analysis = self._analyze_currency_patterns_fanin(account_txns)
        pattern_analysis.update(currency_analysis)

        amount_analysis = self._analyze_amount_patterns(account_txns)
        pattern_analysis.update(amount_analysis)

        if 'Timestamp' in self.df.columns and not self.df['Timestamp'].empty:
            temporal_analysis = self._analyze_temporal_patterns(account_txns)
            pattern_analysis.update(temporal_analysis)
        else:
            pattern_analysis.update({'temporal_clustering_score': 0, 'transaction_velocity': 0})

        # Calculate risk score
        risk_score = self._calculate_fan_risk_score(pattern_analysis)
        pattern_analysis['risk_score'] = risk_score

        if risk_score >= 50:
            return pattern_analysis

        return None

    def _analyze_currency_patterns_fanin(self, transactions):
        """Analyze currency patterns for FAN-IN (group by sender accounts)"""

        try:
            if transactions.empty:
                return {
                    'currency_breakdown': [],
                    'currency_distribution': {},
                    'currency_entropy': 0,
                    'dominant_currency': 'USD',
                    'currency_mixing_ratio': 0
                }

            currency_stats = transactions.groupby('Currency').agg({
                'Amount': ['sum', 'count', 'mean'],
                'Account': 'nunique'  # Count unique senders per currency
            }).reset_index()

            currency_stats.columns = ['Currency', 'total_amount', 'txn_count', 'mean_amount', 'senders']

            # Currency diversity metrics
            currency_distribution = transactions['Currency'].value_counts()

            if len(currency_distribution) > 0:
                currency_entropy = -sum((p/len(transactions)) * np.log2(p/len(transactions))
                                       for p in currency_distribution.values if p > 0)
            else:
                currency_entropy = 0

            return {
                'currency_breakdown': currency_stats.to_dict('records'),
                'currency_distribution': currency_distribution.to_dict(),
                'currency_entropy': currency_entropy,
                'dominant_currency': currency_distribution.index[0] if len(currency_distribution) > 0 else 'USD',
                'currency_mixing_ratio': len(currency_distribution) / len(transactions) if len(transactions) > 0 else 0
            }
        except Exception as e:
            return {
                'currency_breakdown': [],
                'currency_distribution': {},
                'currency_entropy': 0,
                'dominant_currency': 'USD',
                'currency_mixing_ratio': 0
            }

    def _analyze_currency_patterns(self, transactions):
        """Analyze currency patterns for FAN-OUT (group by receiver accounts)"""

        try:
            if transactions.empty:
                return {
                    'currency_breakdown': [],
                    'currency_distribution': {},
                    'currency_entropy': 0,
                    'dominant_currency': 'USD',
                    'currency_mixing_ratio': 0
                }

            currency_stats = transactions.groupby('Currency').agg({
                'Amount': ['sum', 'count', 'mean'],
                'Account.1': 'nunique'  # Count unique recipients per currency
            }).reset_index()

            currency_stats.columns = ['Currency', 'total_amount', 'txn_count', 'mean_amount', 'recipients']

            # Currency diversity metrics
            currency_distribution = transactions['Currency'].value_counts()

            if len(currency_distribution) > 0:
                currency_entropy = -sum((p/len(transactions)) * np.log2(p/len(transactions))
                                       for p in currency_distribution.values if p > 0)
            else:
                currency_entropy = 0

            return {
                'currency_breakdown': currency_stats.to_dict('records'),
                'currency_distribution': currency_distribution.to_dict(),
                'currency_entropy': currency_entropy,
                'dominant_currency': currency_distribution.index[0] if len(currency_distribution) > 0 else 'USD',
                'currency_mixing_ratio': len(currency_distribution) / len(transactions) if len(transactions) > 0 else 0
            }
        except Exception as e:
            return {
                'currency_breakdown': [],
                'currency_distribution': {},
                'currency_entropy': 0,
                'dominant_currency': 'USD',
                'currency_mixing_ratio': 0
            }

    def _analyze_amount_patterns(self, transactions):
        """Analyze amount distribution patterns"""

        if transactions.empty:
            return {
                'min_amount': 0, 'max_amount': 0, 'mean_amount': 0,
                'median_amount': 0, 'std_amount': 0, 'amount_range_ratio': 0,
                'amount_categories': {'micro': 0, 'small': 0, 'medium': 0, 'large': 0, 'mega': 0},
                'round_amount_ratio': 0, 'amount_variety_score': 0
            }

        amounts = transactions['Amount']

        try:
            min_amount = amounts.min() if not amounts.empty else 0
            max_amount = amounts.max() if not amounts.empty else 0
            mean_amount = amounts.mean() if not amounts.empty else 0
            median_amount = amounts.median() if not amounts.empty else 0
            std_amount = amounts.std() if len(amounts) > 1 else 0
            amount_range_ratio = max_amount / max(min_amount, 1e-9) if min_amount > 0 else (max_amount if max_amount > 0 else 0)

            amount_stats = {
                'min_amount': min_amount,
                'max_amount': max_amount,
                'mean_amount': mean_amount,
                'median_amount': median_amount,
                'std_amount': std_amount,
                'amount_range_ratio': amount_range_ratio
            }

            # Amount categorization
            amount_categories = {
                'micro': len(amounts[amounts < 100]),
                'small': len(amounts[(amounts >= 100) & (amounts < 1000)]),
                'medium': len(amounts[(amounts >= 1000) & (amounts < 10000)]),
                'large': len(amounts[(amounts >= 10000) & (amounts < 50000)]),
                'mega': len(amounts[amounts >= 50000])
            }

            # Round number analysis
            round_amounts = sum(1 for amt in amounts if amt % 100 == 0 and amt > 0)
            round_ratio = round_amounts / len(amounts) if len(amounts) > 0 else 0

            amount_stats.update({
                'amount_categories': amount_categories,
                'round_amount_ratio': round_ratio,
                'amount_variety_score': len(set(amounts)) / len(amounts) if len(amounts) > 0 else 0
            })

            return amount_stats
        except Exception as e:
            return {
                'min_amount': 0, 'max_amount': 0, 'mean_amount': 0,
                'median_amount': 0, 'std_amount': 0, 'amount_range_ratio': 0,
                'amount_categories': {'micro': 0, 'small': 0, 'medium': 0, 'large': 0, 'mega': 0},
                'round_amount_ratio': 0, 'amount_variety_score': 0
            }

    def _analyze_temporal_patterns(self, transactions):
        """Analyze temporal coordination patterns"""

        if 'Timestamp' not in transactions.columns or transactions.empty or transactions['Timestamp'].isnull().all():
            return {'temporal_clustering_score': 0, 'transaction_velocity': 0}

        try:
            timestamps = transactions['Timestamp'].dropna().sort_values()

            if len(timestamps) < 2:
                return {'temporal_clustering_score': 0, 'transaction_velocity': len(transactions) / 1.0 if len(transactions) > 0 else 0}

            # Time gap analysis
            time_gaps = timestamps.diff().dt.total_seconds().dropna()

            # Clustering analysis
            rapid_sequences = sum(1 for gap in time_gaps if gap <= 60)    # Within 1 minute
            hourly_sequences = sum(1 for gap in time_gaps if gap <= 3600) # Within 1 hour
            daily_sequences = sum(1 for gap in time_gaps if gap <= 86400) # Within 1 day

            # Velocity analysis
            total_time_seconds = (timestamps.max() - timestamps.min()).total_seconds()
            total_time_hours = total_time_seconds / 3600 if total_time_seconds > 0 else 0
            transaction_velocity = len(transactions) / max(total_time_hours, 1) if total_time_hours > 0 else 0

            return {
                'rapid_sequences': rapid_sequences,
                'hourly_sequences': hourly_sequences,
                'daily_sequences': daily_sequences,
                'transaction_velocity': transaction_velocity,
                'temporal_clustering_score': (rapid_sequences * 3 + hourly_sequences * 2 + daily_sequences) / len(transactions) if len(transactions) > 0 else 0
            }
        except Exception as e:
            return {'temporal_clustering_score': 0, 'transaction_velocity': 0}

    def _calculate_fan_risk_score(self, analysis):
        """Calculate risk score for both FAN-OUT and FAN-IN patterns"""

        risk_score = 0

        # Factor 1: Fan degree (connections count)
        fan_degree = analysis['fan_degree']
        for threshold, points in sorted(self.risk_matrix['fan_thresholds'].items()):
            if fan_degree >= threshold:
                risk_score += points

        # Factor 2: Currency diversity
        currencies = analysis['unique_currencies']
        for threshold, points in sorted(self.risk_matrix['currency_thresholds'].items()):
            if currencies >= threshold:
                risk_score += points

        # Factor 3: Total amount
        total_amount = analysis['total_amount']
        for threshold, points in sorted(self.risk_matrix['amount_thresholds'].items()):
            if total_amount >= threshold:
                risk_score += points

        # Factor 4: Time coordination
        if 'time_span_days' in analysis and analysis['time_span_days'] is not None:
            time_span = analysis['time_span_days']
            for threshold, points in sorted(self.risk_matrix['time_thresholds'].items()):
                if time_span <= threshold and time_span >= 0:
                    risk_score += points
                    break

        # Bonus factors
        if analysis.get('amount_variety_score', 0) > 0.8:
            risk_score += 15
        elif analysis.get('amount_variety_score', 0) > 0.6:
            risk_score += 10

        if analysis.get('currency_mixing_ratio', 0) > 0.5:
            risk_score += 20
        elif analysis.get('currency_mixing_ratio', 0) > 0.3:
            risk_score += 15

        temporal_score = analysis.get('temporal_clustering_score', 0)
        if temporal_score > 1.0:
            risk_score += 15
        elif temporal_score > 0.5:
            risk_score += 10

        if analysis.get('amount_range_ratio', 1) > 100:
            risk_score += 10

        return min(risk_score, 100)

    def display_all_patterns(self, top_k_each=5, show_details=True):
        """Display both FAN-OUT and FAN-IN patterns"""

        print(f"\n🚨 DETECTED MONEY LAUNDERING PATTERNS")
        print("="*80)

        total_patterns = len(self.detected_fanout_patterns) + len(self.detected_fanin_patterns)
        if total_patterns == 0:
            print("❌ No suspicious patterns detected")
            print("💡 Try adjusting parameters:")
            print("   - Lower min_connections to 2")
            print("   - Lower min_total_amount to 5000")
            return

        print(f"Total patterns found: {total_patterns}")
        print(f"📤 FAN-OUT patterns: {len(self.detected_fanout_patterns)}")
        print(f"📥 FAN-IN patterns: {len(self.detected_fanin_patterns)}")

        # Display FAN-OUT patterns
        if self.detected_fanout_patterns:
            print(f"\n📤 FAN-OUT PATTERNS (One sender → Many receivers)")
            print("="*60)
            self._display_patterns(self.detected_fanout_patterns[:top_k_each], show_details)

        # Display FAN-IN patterns
        if self.detected_fanin_patterns:
            print(f"\n📥 FAN-IN PATTERNS (Many senders → One receiver)")
            print("="*60)
            self._display_patterns(self.detected_fanin_patterns[:top_k_each], show_details)

    def _display_patterns(self, patterns, show_details=True):
        """Display pattern details"""

        for i, pattern in enumerate(patterns, 1):
            risk_level = self._get_risk_level(pattern['risk_score'])
            pattern_type = pattern['pattern_type']

            print(f"\n🔥 PATTERN #{i}: {pattern_type} - {risk_level} RISK")
            print(f"🎯 Account: {pattern['account']}")
            print(f"⚠️ Risk Score: {pattern['risk_score']:.1f}/100")
            print(f"🌟 {pattern_type} Degree: {pattern['fan_degree']} {pattern['connection_type']}")
            print(f"💱 Currencies: {pattern['unique_currencies']} different")
            print(f"💰 Total Amount: ${pattern['total_amount']:,.2f}")
            print(f"⏱️ Time Span: {pattern['time_span_days']} days" if pattern['time_span_days'] is not None else "⏱️ Time Span: N/A")
            print(f"📊 Transactions: {pattern['transaction_count']}")

            if show_details:
                self._display_pattern_details(pattern)

            print("-" * 60)

    def _display_pattern_details(self, pattern):
        """Display detailed pattern analysis"""

        print(f"\n💱 CURRENCY BREAKDOWN:")
        for currency_info in pattern['currency_breakdown']:
            currency = currency_info['Currency']
            amount = currency_info['total_amount']
            txns = currency_info['txn_count']

            if pattern['pattern_type'] == 'FAN-OUT':
                connections = currency_info.get('recipients', 0)
                conn_type = "recipients"
            else:
                connections = currency_info.get('senders', 0)
                conn_type = "senders"

            print(f"   • {currency}: ${amount:,.2f} ({txns} txns from/to {connections} {conn_type})")

        print(f"\n💰 AMOUNT ANALYSIS:")
        categories = pattern['amount_categories']
        print(f"   • Micro (<$100): {categories['micro']} transactions")
        print(f"   • Small ($100-$1K): {categories['small']} transactions")
        print(f"   • Medium ($1K-$10K): {categories['medium']} transactions")
        print(f"   • Large ($10K-$50K): {categories['large']} transactions")
        print(f"   • Mega (>$50K): {categories['mega']} transactions")

        print(f"   • Amount Range: ${pattern['min_amount']:,.2f} - ${pattern['max_amount']:,.2f}")
        print(f"   • Range Ratio: {pattern['amount_range_ratio']:,.1f}x")

        print(f"\n💸 SAMPLE TRANSACTIONS (First 3):")
        sample_txns = pattern['transactions'].head(3)
        for _, txn in sample_txns.iterrows():
            if pattern['pattern_type'] == 'FAN-OUT':
                direction = f"{txn['Account']} → {txn['Account.1']}"
            else:
                direction = f"{txn['Account']} → {txn['Account.1']}"

            amount = txn['Amount']
            currency = txn.get('Currency', 'USD')
            timestamp = txn.get('Timestamp', 'N/A')
            print(f"   💸 {direction}: {amount:,.2f} {currency} ({timestamp})")

    def _get_risk_level(self, score):
        """Get risk level description"""
        if score >= 85:
            return "🚨 CRITICAL"
        elif score >= 70:
            return "🔴 HIGH"
        elif score >= 60:
            return "🟠 MEDIUM"
        else:
            return "🟡 LOW"

    def export_all_results(self, filename_prefix='fan_pattern_results'):
        """Export both FAN-OUT and FAN-IN results"""

        # Export FAN-OUT patterns
        if self.detected_fanout_patterns:
            fanout_data = []
            for pattern in self.detected_fanout_patterns:
                fanout_data.append({
                    'Pattern_Type': 'FAN-OUT',
                    'Account': pattern['account'],
                    'Risk_Score': pattern['risk_score'],
                    'Fan_Degree': pattern['fan_degree'],
                    'Connection_Type': pattern['connection_type'],
                    'Unique_Currencies': pattern['unique_currencies'],
                    'Total_Amount': pattern['total_amount'],
                    'Time_Span_Days': pattern['time_span_days'] if pattern['time_span_days'] is not None else 'N/A',
                    'Transaction_Count': pattern['transaction_count'],
                    'Risk_Level': self._get_risk_level(pattern['risk_score']).replace('🚨 ', '').replace('🔴 ', '').replace('🟠 ', '').replace('🟡 ', ''),
                    'Amount_Range_Ratio': pattern.get('amount_range_ratio', 0),
                    'Temporal_Clustering': pattern.get('temporal_clustering_score', 0)
                })

            fanout_df = pd.DataFrame(fanout_data)
            fanout_filename = f'{filename_prefix}_fanout.csv'
            fanout_df.to_csv(fanout_filename, index=False)
            print(f"📁 Exported {len(fanout_data)} FAN-OUT patterns to {fanout_filename}")

        # Export FAN-IN patterns
        if self.detected_fanin_patterns:
            fanin_data = []
            for pattern in self.detected_fanin_patterns:
                fanin_data.append({
                    'Pattern_Type': 'FAN-IN',
                    'Account': pattern['account'],
                    'Risk_Score': pattern['risk_score'],
                    'Fan_Degree': pattern['fan_degree'],
                    'Connection_Type': pattern['connection_type'],
                    'Unique_Currencies': pattern['unique_currencies'],
                    'Total_Amount': pattern['total_amount'],
                    'Time_Span_Days': pattern['time_span_days'] if pattern['time_span_days'] is not None else 'N/A',
                    'Transaction_Count': pattern['transaction_count'],
                    'Risk_Level': self._get_risk_level(pattern['risk_score']).replace('🚨 ', '').replace('🔴 ', '').replace('🟠 ', '').replace('🟡 ', ''),
                    'Amount_Range_Ratio': pattern.get('amount_range_ratio', 0),
                    'Temporal_Clustering': pattern.get('temporal_clustering_score', 0)
                })

            fanin_df = pd.DataFrame(fanin_data)
            fanin_filename = f'{filename_prefix}_fanin.csv'
            fanin_df.to_csv(fanin_filename, index=False)
            print(f"📁 Exported {len(fanin_data)} FAN-IN patterns to {fanin_filename}")

        # Export combined results
        if self.detected_fanout_patterns or self.detected_fanin_patterns:
            all_data = fanout_data + fanin_data if self.detected_fanout_patterns and self.detected_fanin_patterns else (fanout_data if self.detected_fanout_patterns else fanin_data)
            all_df = pd.DataFrame(all_data)
            export_path = "/home/sujeet/work/Anti Money Laundering/riskguard-dash-main/public"
            combined_filename = os.path.join(export_path, f"{filename_prefix}_combined.csv")

            #combined_filename = f'{filename_prefix}_combined.csv'
            all_df.to_csv(combined_filename, index=False)
            print(f"📁 Exported {len(all_data)} total patterns to {combined_filename}")

# Main execution functions
def main():
    """Execute both FAN-OUT and FAN-IN detection"""
    print("🎯 ADVANCED FAN-OUT & FAN-IN MONEY LAUNDERING DETECTION")
    print("="*75)
    print("📤 FAN-OUT: Distribution patterns (1 → Many)")
    print("📥 FAN-IN: Consolidation patterns (Many → 1)")

    # Initialize detector
    detector = AdvancedFanDetector('env_name/LI-Small_Trans.csv')

    # Detect both patterns
    fanout_patterns, fanin_patterns = detector.detect_all_fan_patterns(
        min_connections=3,
        max_time_window_days=30,
        min_total_amount=10000,
        max_accounts_to_analyze=100
    )

    # Display results
    detector.display_all_patterns(top_k_each=5, show_details=True)

    # Show summary
    total_patterns = len(fanout_patterns) + len(fanin_patterns)
    if total_patterns > 0:
        print(f"\n📊 DETECTION SUMMARY")
        print("="*40)
        print(f"📤 FAN-OUT Patterns: {len(fanout_patterns)}")
        print(f"📥 FAN-IN Patterns: {len(fanin_patterns)}")
        print(f"🎯 Total Patterns: {total_patterns}")

        if fanout_patterns:
            avg_fanout_degree = np.mean([p['fan_degree'] for p in fanout_patterns])
            print(f"📊 Avg FAN-OUT degree: {avg_fanout_degree:.1f}")

        if fanin_patterns:
            avg_fanin_degree = np.mean([p['fan_degree'] for p in fanin_patterns])
            print(f"📊 Avg FAN-IN degree: {avg_fanin_degree:.1f}")

    # Export results
    if total_patterns > 0:
        detector.export_all_results()

    return fanout_patterns, fanin_patterns

def quick_analysis():
    """Quick analysis with lower thresholds"""
    detector = AdvancedFanDetector('env_name/LI-Small_Trans.csv')

    fanout_patterns, fanin_patterns = detector.detect_all_fan_patterns(
        min_connections=2,
        max_time_window_days=30,
        min_total_amount=10000,
        max_accounts_to_analyze=50
    )

    detector.display_all_patterns(top_k_each=3, show_details=False)
    return fanout_patterns, fanin_patterns

if __name__ == "__main__":
    # Run comprehensive analysis
    results = main()

    # If no results, try quick analysis
    if not results[0] and not results[1]:
        print("\n🔄 No patterns with strict criteria. Trying relaxed analysis...")
        quick_results = quick_analysis()