import itertools
import pandas as pd


class Diff(object):
    def __init__(self, lhs, rhs, on, lhs_name="LHS", rhs_name="RHS",
                 comparison_col_name="COMPARISON_COL", *args, **kwargs):
        self.lhs = lhs.copy(deep=True)
        self.rhs = rhs.copy(deep=True)
        self.input_key = on
        self.key = self.input_key + [comparison_col_name]
        self.lhs_name = lhs_name
        self.rhs_name = rhs_name
        self.diff_key = ["DIFF", "ABS_DIFF", "PCT_DIFF", "ABS_PCT_DIFF"]
        self.output_key = self.key + [self.lhs_name,
                                      self.rhs_name] + self.diff_key

        if self.lhs.index.name != self.input_key:
            self.lhs.set_index(self.input_key, inplace=True)

        if self.rhs.index.name != self.input_key:
            self.rhs.set_index(self.input_key, inplace=True)

        self._set_value_cols()

        self.df = self.construct_diff_df()

    def __str__(self):
        return self.df.to_string()

    def __repr__(self):
        return self.df.to_string()

    def _repr_html_(self):
        # this is for displaying in IPython
        return self.df._repr_html_()

    def _set_value_cols(self):
        self.value = list(set(self.lhs.columns).intersection(
            self.rhs.columns))

    def _construct_master_index_df(self):
        """Helper function to construct unique index for the diff dataframe.
        """
        unique_idx = set(self.lhs.index.tolist() + self.rhs.index.tolist())
        value_cols = list(set(self.lhs.columns).difference(self.input_key))
        if len(self.input_key) == 1:
            x_prod_idx = tuple(list(itertools.product(unique_idx, value_cols)))
        else:
            x_prod_idx = tuple(list(k[0]) + [k[1]] for k in
                               list(itertools.product(unique_idx, value_cols)))
        df = pd.MultiIndex.from_tuples(
            x_prod_idx, names=self.key).to_frame()
        df = df.sort_index()  # speed up lex sort
        df = df.reset_index(drop=True)
        return df

    def _separate_duplicates(self, df, key, series_name):
        df = df.stack().reset_index()
        df.columns = key + [series_name]
        return df.drop_duplicates(), df[df.duplicated()]

    def _split_out_numeric_values(self, df):
        df["IS_LHS_NUMERIC"] = pd.to_numeric(df[self.lhs_name], errors="coerce")
        df["IS_RHS_NUMERIC"] = pd.to_numeric(df[self.rhs_name], errors="coerce")

        numeric = df["IS_LHS_NUMERIC"].notnull() | df[
            "IS_RHS_NUMERIC"].notnull()
        index_cols = self.key + ["IS_DUPLICATED"]
        value_cols = [self.lhs_name, self.rhs_name]

        df_numeric = df[numeric][index_cols + value_cols]
        for c in value_cols:
            df_numeric[c] = pd.to_numeric(df_numeric[c])

        df_non_numeric = df[~numeric][index_cols + value_cols]

        return df_numeric, df_non_numeric

    def _diff_numeric(self, df):
        """Calculate differences for numeric values

        Do not fill missing values; allow the differences to have nulls.

        """
        not_null = df[self.lhs_name].notnull() & df[self.rhs_name].notnull()
        df.loc[not_null, "DIFF"] = df.loc[not_null, self.lhs_name] - \
                                   df.loc[not_null, self.rhs_name]
        df.loc[not_null, "ABS_DIFF"] = df.loc[not_null, "DIFF"].abs()

        # PCT_DIFF = RHS / LHS - 1
        df.loc[not_null, "PCT_DIFF"] = df.loc[not_null, self.rhs_name] / \
                                       df.loc[not_null, self.lhs_name] - 1
        df.loc[not_null, "ABS_PCT_DIFF"] = df.loc[not_null, "PCT_DIFF"].abs()

        # remove all records with no differences
        df = df.loc[df["ABS_PCT_DIFF"] > 0]
        return df

    def _diff_non_numeric(self, df):
        """Display differences for non-numeric values.

        TODO: maybe calculate string difference here

        """
        for c in self.diff_key:
            df.loc[df[self.lhs_name] != df[self.rhs_name],
                   c] = df[self.lhs_name].fillna("MISSING") + " <> " + \
                        df[self.rhs_name].fillna("MISSING")

        # remove all records with no differences
        df = df.loc[df["ABS_PCT_DIFF"].notnull()]
        return df

    def _generate_diffs(self, df):
        df.dropna(subset=[self.lhs_name, self.rhs_name],
                  how="all", inplace=True)
        numeric, non_numeric = self._split_out_numeric_values(df)

        numeric = self._diff_numeric(df=numeric)
        numeric = numeric.sort_values(["ABS_PCT_DIFF"], ascending=False)
        non_numeric = self._diff_non_numeric(df=non_numeric)

        df = pd.concat([numeric, non_numeric], ignore_index=True)
        return df

    def construct_diff_df(self):
        master_index_df = self._construct_master_index_df()
        lhs, lhs_duplicates = self._separate_duplicates(
            df=self.lhs, key=self.key, series_name="LHS")
        rhs, rhs_duplicates = self._separate_duplicates(
            df=self.rhs, key=self.key, series_name="RHS")

        # left join twice is more efficient than an outer join
        df = pd.merge(left=master_index_df, right=lhs, how="left", on=self.key)
        df = pd.merge(left=df, right=rhs, how="left", on=self.key)

        # flag duplicates
        df["IS_DUPLICATED"] = 0
        lhs_duplicates["IS_DUPLICATED"] = 1
        rhs_duplicates["IS_DUPLICATED"] = 1
        df = pd.concat([df, lhs_duplicates, rhs_duplicates],
                       ignore_index=True, sort=True)

        df = self._generate_diffs(df)
        for c in ["DIFF", "ABS_DIFF", "PCT_DIFF", "ABS_PCT_DIFF"]:
            df.loc[df["IS_DUPLICATED"] & df[
                self.lhs_name].notnull(), c] = f"DUPLICATED_ON_{self.lhs_name}"
            df.loc[df["IS_DUPLICATED"] & df[
                self.rhs_name].notnull(), c] = f"DUPLICATED_ON_{self.rhs_name}"

        return df[self.output_key].set_index(self.input_key)

    def set_index(self):
        if len(self.input_key) == 1:
            if self.df.index.name != self.input_key[0]:
                self.df.set_index(self.input_key, inplace=True)
        else:
            if self.df.index.names != self.input_key:
                self.df.set_index(self.input_key, inplace=True)

    def reset_index(self):
        self.df.reset_index(inplace=True)
