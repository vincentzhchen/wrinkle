import itertools
import pandas as pd


class Diff(object):
    def __init__(self, lhs, rhs, on, lhs_name="LHS", rhs_name="RHS",
                 comparison_col_name="COMPARISON_COL", *args, **kwargs):

        # setup LHS and RHS dataframes
        self.lhs = lhs.copy(deep=True)
        self.rhs = rhs.copy(deep=True)
        if self.lhs.index.name != on:
            self.lhs.set_index(on, inplace=True)

        if self.rhs.index.name != on:
            self.rhs.set_index(on, inplace=True)

        # setup keys and values
        self.input_key = on
        self.key = self.input_key + [comparison_col_name]
        self.lhs_name = lhs_name
        self.rhs_name = rhs_name
        self.diff_key = ["DIFF", "ABS_DIFF", "PCT_DIFF", "ABS_PCT_DIFF"]
        self.output_key = self.key + [lhs_name, rhs_name] + self.diff_key
        self.value = list(set(self.lhs.columns).intersection(
            self.rhs.columns))

        # create diff on instantiation
        self.df = self.construct_diff_df()

    def __str__(self):
        return self.df.to_string()

    def __repr__(self):
        return self.df.to_string()

    def _repr_html_(self):
        # this is for displaying in IPython
        return self.df._repr_html_()

    def to_frame(self):
        return self.df

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

        # since NaN was not filled, ABS_PCT_DIFF has nulls -- mark with -1
        df["ABS_PCT_DIFF"].fillna(-1, inplace=True)

        # remove all records with no differences
        df = df.loc[df["ABS_PCT_DIFF"] != 0]  # this keeps the NaN comparisons

        df = df.sort_values(["ABS_PCT_DIFF"], ascending=False)
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
        # NaN != NaN so if both sides are NaN, drop the record
        df.dropna(subset=[self.lhs_name, self.rhs_name],
                  how="all", inplace=True)

        # handle numeric values and strings differently
        numeric, non_numeric = self._split_out_numeric_values(df)
        numeric = self._diff_numeric(df=numeric)
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

    def to_excel(self, path, reset_index=True):
        writer = pd.ExcelWriter(path=path,
                                date_format="YYYY-MM-DD",
                                datetime_format="YYYY-MM-DD")

        df = self.df
        lhs = self.lhs
        rhs = self.rhs

        if reset_index:
            df = self.df.reset_index()
            lhs = self.lhs.reset_index()
            rhs = self.rhs.reset_index()

        df.to_excel(writer, sheet_name="DIFF", index=False)
        lhs.to_excel(writer, sheet_name=self.lhs_name, index=False)
        rhs.to_excel(writer, sheet_name=self.rhs_name, index=False)
        writer.save()
