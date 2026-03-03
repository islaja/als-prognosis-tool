from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class AutoColumnPreprocess(BaseEstimator, TransformerMixin):

    def __init__(self, scaler_class="StandardScaler"):
        self.preprocessor = None
        self.scaler_class = scaler_class

    def _get_scaler_instance(self, scaler_class):
        if scaler_class.lower() == "standardscaler":
            return StandardScaler()
        elif scaler_class.lower() == "robustscaler":
            return RobustScaler()
        elif scaler_class.lower() == "minmaxscaler":
            return MinMaxScaler()
        elif scaler_class.lower() == "none":
            return None
        else:
            raise ValueError(f"Unknown scaler class: {scaler_class}")
        
    def identify_categorical_types(self, data):
        import pandas as pd
        """
        Identifies nominal and ordinal categorical features in a DataFrame.

        Args:
            data (pd.DataFrame): Input DataFrame containing potential categorical features.

        Returns:
            tuple: A tuple containing three lists:
                - categorical_cols (list): List of all identified categorical columns.
                - nominal_cols (list): List of identified nominal categorical columns.
                - ordinal_cols (list): List of identified ordinal categorical columns.
        """

        categorical_cols = data.select_dtypes(include=['category']).columns.tolist()
        nominal_cols = []
        ordinal_cols = []

        for col in categorical_cols:
            if not pd.api.types.is_categorical_dtype(data[col]):
            # Not actually categorical, skip
                continue
            if data[col].dtype.ordered:
                ordinal_cols.append(col)
            else:
                nominal_cols.append(col)

        return categorical_cols, nominal_cols, ordinal_cols
        
    def fit(self, X, y=None):
        self.scaler = self._get_scaler_instance(self.scaler_class)  # Dynamic instantiation
        
        numeric_cols = X.select_dtypes(exclude=['category']).columns.tolist()

        _, nominal_categorical_cols, ordinal_categorical_cols = self.identify_categorical_types(X.copy())

        numeric_transformer_steps = [
            ('imputer', SimpleImputer(strategy='mean')),
        ]
      
        if self.scaler is not None:
            numeric_transformer_steps.append(('scaler', self.scaler))

        numeric_transformer = Pipeline(steps=numeric_transformer_steps)

        nominal_categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore'))
            ])
        ordinal_categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])

        self.preprocessor = ColumnTransformer([
                ("numeric", numeric_transformer, numeric_cols),
                ("nominal_categorical", nominal_categorical_transformer, nominal_categorical_cols),
                ("ordinal_categorical", ordinal_categorical_transformer, ordinal_categorical_cols)],
                verbose_feature_names_out=False, verbose=0)

        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            self.fit(X)
   
        return self.preprocessor.transform(X)
        
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self):
        # Get feature names from the final transformer in the ColumnTransformer
        return self.preprocessor.get_feature_names_out()