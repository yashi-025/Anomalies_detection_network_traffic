import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class KDDDataLoader:
    def __init__(self, data_path, names_file, is_labeled=True, columns_to_keep = None):
        self.data_path = data_path
        self.names_file = names_file
        self.is_labeled = is_labeled
        self.columns = self._extract_column_names()
        self.df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.columns_to_keep = columns_to_keep  # Track columns after processing

    def _extract_column_names(self):
        # def _extract_column_names(self):
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
            'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
            'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate'
        ]
        if self.is_labeled:
            columns.append("label")
        return columns


    def load_data(self):
        self.df = pd.read_csv(self.data_path, names=self.columns, header=None)

        # Convert labels to binary
        if self.is_labeled:
            self.df['label'] = self.df['label'].apply(lambda x: 0 if x.strip() == 'normal.' else 1)
        return self.df

    def encode_categorical(self):
        cat_columns = self.df.select_dtypes(include=['object']).columns
        for col in cat_columns:
            if col != 'label':
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
        return self.df

    def normalize_numerical(self):
        # Separate label
        label_col = self.df['label'] if self.is_labeled else None
        features = self.df.drop(columns=['label']) if self.is_labeled else self.df

        num_columns = features.select_dtypes(include=['int64', 'float64']).columns
        features[num_columns] = self.scaler.fit_transform(features[num_columns])

        # Drop inf, NaN, constant columns
        features = features.replace([float('inf'), float('-inf')], pd.NA).dropna()
        features = features.loc[:, features.std(numeric_only=True) != 0]
        
        # SAVE the final column list after processing (only during training)
        if self.columns_to_keep is None:
            self.columns_to_keep = features.columns.tolist()
        else:
            # Use existing training columns to align test features
            features = features.reindex(columns=self.columns_to_keep, fill_value=0)

        if self.is_labeled:
            # Reattach label (reset index to match)
            label_col = label_col.loc[features.index]
            features['label'] = label_col

        self.df = features
        return self.df

    def preprocess(self):
        print("Loading and preprocessing data...")
        self.load_data()
        self.encode_categorical()
        self.normalize_numerical()
        return self.df

    def get_processed_data(self):
        return self.df
