# import library
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from scipy.stats import t
from matplotlib.lines import Line2D
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Define the clear
def clear():
    os.system('cls')

# Define the menu
def read_data():
    path = input("[>] Enter the Excel file path (for example: data.xlsx): ")
    try:
        data = pd.read_excel(path)
        clear()
        print("[=] Data read successfully! [=]")
        return data
    except Exception as e:
        clear()
        print(f"[x] Data read not successfully! [x]")
        return None

def show_data(data):
    if data is not None:
        pd.set_option('display.max_rows', None)  
        pd.set_option('display.max_columns', None)  
        pd.set_option('display.width', None)  
        pd.set_option('display.max_colwidth', None)
        clear() 
        print("\n==================================")
        print("|                                |")
        print("|      DISPLAY OF INPUT DATA     |")
        print("|                                |")
        print("=================================\n")
        print(data)
    else:
        clear()
        print("[x] There is no data available. Please read the data first. [x]")

def onewayANOVA(data):
    if data is not None:
        clear()
        print("\n==================================")
        print("|                                |")
        print("|    DISPLAY OF ONE WAY ANOVA    |")
        print("|                                |")
        print("=================================\n")
        print("\n[>] Available columns: ", list(data.columns))
        responses = input("[1] Enter the response column: ")
        factors = input("[2] Enter a factor column: ")
        try:
            alpha = float(input("[3] Enter the significance level (example: 0.05): "))
            if not (0 < alpha < 1):
                raise ValueError("The significance level should be between 0 and 1.")
        except ValueError as e:
            print(f"[x] {e} [x]")
            return

        if responses in data.columns and factors in data.columns:
            formula = f"{responses} ~ C({factors})"
            model = ols(formula, data=data).fit()
            anovaTable = sm.stats.anova_lm(model, typ=2)
            anovaTable["Mean Squares"] = anovaTable["Sum of Squares"] / anovaTable["df"]
            SSA = anovaTable.loc[f"C({factors})", "Sum of Squares"]
            SSW = anovaTable.loc["Residual", "Sum of Squares"]
            SST = SSA + SSW
            factorInfo = data[factors].value_counts().reset_index()
            factorInfo.columns = ["Factor Levels", "Counts"]
            groupStats = data.groupby(factors)[responses].agg(N="count", Mean="mean", StDev="std").reset_index()
            groupStats["MOE"] = groupStats.apply(lambda row: t.ppf(1 - alpha / 2, df=row["N"] - 1) * (row["StDev"] / (row["N"]**0.5)),axis=1)
            groupStats["CI Lower"] = groupStats["Mean"] - groupStats["MOE"]
            groupStats["CI Upper"] = groupStats["Mean"] + groupStats["MOE"]

            # Result of One-Way ANOVA
            clear()
            print("\n==================================")
            print("|                                |")
            print("|    RESULTS OF ONE WAY ANOVA    |")
            print("|                                |")
            print("=================================\n")
            print(f"Method: One-Way ANOVA")
            print(f"(H0): There is no difference in means between groups.")
            print(f"(H1): There is at least one group that is different.")
            print(f"Significance Level: {alpha}")

            print("\n[=] Factor Information: [=]")
            print(factorInfo.toString(index=False))

            print("\n[=] One-Way ANOVA Summary Table [=]")
            print(f"Term            Sum of Squares           df            Mean Squares             F       P-value")
            print(f"SSA             {SSA:.4f}          {anovaTable['df'].iloc[0]:.0f}          {anovaTable['Mean Squares'].iloc[0]:.4f}     {anovaTable['F'].iloc[0]:.4f}     {anovaTable['PR(>F)'].iloc[0]:.4f}")
            print(f"SSW (Residual)  {SSW:.4f}          {anovaTable['df'].iloc[1]:.0f}          {anovaTable['Mean Squares'].iloc[1]:.4f}")
            print(f"SST (Total)     {SST:.4f}          {anovaTable['df'].iloc[0] + anovaTable['df'].iloc[1]:.0f}")

            print("\n[=] Means [=]")
            print(groupStats.toString(index=False))

            # Post-Hoc Test (Tukey)
            tukey = pairwise_tukeyhsd(endog=data[responses], groups=data[factors], alpha=alpha)
            print("\n[=] Post-Hoc Test (Tukey) [=]")
            print(tukey)

            # Conclusion
            f_test = anovaTable['F'].iloc[0]
            f_table = stats.f.ppf(1 - alpha, dfn=anovaTable['df'].iloc[0], dfd=anovaTable['df'].iloc[1])
            print("\n[=] Decision [=]")
            if f_test > f_table:
                print(f"Reject H0: There is a significant difference between the groups (F_test = {f_test:.3f}, F_table = {f_table:.3f}).")
            else:
                print(f"Failure to reject H0: There is no significant difference between the groups (F_test = {f_test:.3f}, F_table = {f_table:.3f}).")
        else:
            clear()
            print("[x] Invalid column. Please try again. [x]")
    else:
        clear()
        print("[x] No data is available. Please load the data first. [x]")


def twowayANOVA(data):
    if data is not None:
        clear()
        print("\n==================================")
        print("|                                |")
        print("|    DISPLAY OF TWO WAY ANOVA    |")
        print("|                                |")
        print("=================================\n")
        print("\n[>] Available columns: ", list(data.columns))

        responses2 = input("[1] Enter the responses column (separate with commas): ").strip().split(',')
        responses2 = [col.strip() for col in responses2]
        factors2 = input("[2] Enter a factors column (separate with commas): ").strip().split(',')
        factors2 = [col.strip() for col in factors2]
        
        if not all(col in data.columns for col in factors2 + responses2):
            print("[x] Some columns are invalid. Please check your input. [x]")
            return
        alpha = float(input("[3] Enter the significance level (example: 0.05): "))
        
        # Result of Two-Way ANOVA
        clear()
        print("\n==================================")
        print("|                                |")
        print("|    RESULTS OF TWO WAY ANOVA    |")
        print("|                                |")
        print("=================================\n")
        print("\n[=] Factor Information [=]")
        for factor in factors2:
            print(f"{'Factor:':<15} {factor:<20} {'Levels:':<8} {data[factor].nunique():<4} {'Values:':<8} {', '.join(map(str, data[factor].unique()))}")
        
        for response in responses2:
            print(f"[=] Analysis for response {response} [=]")
            formula = f"{response} ~ " + " + ".join([f"C({factor})" for factor in factors2]) + " + " + \
                      " * ".join([f"C({factor})" for factor in factors2])
            model = ols(formula, data=data).fit()
            anovaTable = sm.stats.anova_lm(model, typ=2)

            SSA = anovaTable.loc['C(' + factors2[0] + ')', 'Sum of Squares']
            SSB = anovaTable.loc['C(' + factors2[1] + ')', 'Sum of Squares']
            SSAB = anovaTable.loc['C(' + factors2[0] + '):C(' + factors2[1] + ')', 'Sum of Squares']
            SSW = anovaTable.loc['Residual', 'Sum of Squares']
            SST = SSA + SSB + SSAB + SSW

            a = data[factors2[0]].nunique()
            b = data[factors2[1]].nunique()
            n = len(data) / (a * b)
            
            df_ab = (a - 1) * (b - 1)
            df_error = a * b * (n - 1)
            df_total = a * b * n - 1
            
            print("\n[=] Two-Way ANOVA Summary Table [=]")
            print(f"{'Source':<20} {'DF':<10} {'Adj SS':<10} {'Adj MS':<10} {'F-Value':<10} {'P-Value':<10}")
            print(f"{'SSA':<20} {a - 1:<10} {SSA:<10.4f} {SSA/(a - 1):<10.4f} {SSA/(a - 1)/(SSW/df_error):<10.4f} {anovaTable.loc['C(' + factors2[0] + ')', 'PR(>F)']:.4f}")
            print(f"{'SSB':<20} {b - 1:<10} {SSB:<10.4f} {SSB/(b - 1):<10.4f} {SSB/(b - 1)/(SSW/df_error):<10.4f} {anovaTable.loc['C(' + factors2[1] + ')', 'PR(>F)']:.4f}")
            print(f"{'Interaction AB':<20} {df_ab:<10} {SSAB:<10.4f} {SSAB/df_ab:<10.4f} {SSAB/df_ab/(SSW/df_error):<10.4f} {anovaTable.loc['C(' + factors2[0] + '):C(' + factors2[1] + ')', 'PR(>F)']:.4f}")
            print(f"{'SSW (Residuals)':<20} {df_error:<10} {SSW:<10.4f} {SSW/df_error:<10.4f} {'-':<10}")
            print(f"{'SST (Total)':<20} {df_total:<10} {SST:<10.4f} {SST/df_total:<10.4f} {'-':<10}")

            mse = anovaTable.loc["Residuals", "Sum of Squares"] / anovaTable.loc["Residuals", "df"]
            s = mse**0.5
            ssTotal = anovaTable["Sum of Squares"].sum()
            ssModal = ssTotal - anovaTable.loc["Residuals", "Sum of Squares"]
            rSq = ssModal / ssTotal
            rSq_adj = 1 - (1 - rSq) * ((len(data) - 1) / (len(data) - len(model.params)))
            
            print("\n[=] Model Summary [=]")
            print(f"{'Standard Error (S):':<25} {s:.4f}")
            print(f"{'R-squared:':<25} {rSq:.4f}")
            print(f"{'Adjusted R-squared:':<25} {rSq_adj:.4f}")
            
            print("\n[=] Coefficients [=]")
            print(f"{'Term':<25} {'Coef':<10} {'SE Coef':<10} {'T-Value':<10} {'P-Value':<10} {'VIF':<10}")
            for term, coef, se_coef, t_val, p_val in zip(model.params.index, model.params.values, 
                                                         model.bse.values, model.tvalues.values, 
                                                         model.pvalues.values):
                vif = 1 / (1 - model.rsquared)
                print(f"{term:<25} {coef:<10.4f} {se_coef:<10.4f} {t_val:<10.4f} {p_val:<10.4f} {vif:<10.4f}")

            print("\n[=] Decision [=]")
            for idx, row in anovaTable.iterrows():
                f_val = row["F"]
                p_val = row["PR(>F)"]
                if p_val < alpha:
                    print(f"Reject H0 for {idx}: There is a significant difference (F = {f_val:.3f}, P = {p_val:.3f})")
                else:
                    print(f"Fail to reject H0 for {idx}: No significant difference (F = {f_val:.3f}, P = {p_val:.3f})")
    else:
        clear()
        print("[x] No data is available. Please load the data first. [x]")


def summary_data(data):
    if data is not None:
        clear()
        print("\n==================================")
        print("|                                |")
        print("|          SUMMARY DATA          |")
        print("|                                |")
        print("=================================\n")
        print("[>] Number of Rows: ", data.shape[0])
        print("[>] Number of Columns: ", data.shape[1])
        print("[>] Columns: ", list(data.columns))
        print("\n[=] Statistical Description [=]")
        print(data.describe())
    else:
        clear()
        print("[x] No data is available. Please load the data first. [x]")


def plot_data(data):
    if data is not None:
        clear()
        print("\n==================================")
        print("|                                |")
        print("|          SUMMARY DATA          |")
        print("|                                |")
        print("=================================\n")
        print("[>] Columns: ", list(data.columns))

        factors = input("[1] Enter the factor column for the X axis : ")
        if factors not in data.columns:
            print(f"[x] The column ‘{factors}’ does not exist in the data. [x]")
            return
        numericCols = data.select_dtypes(include=[np.number]).columns.tolist()
        if factors in numericCols:
            numericCols.remove(factors)
        meltedData = data.melt(id_vars=[factors], value_vars=numericCols, var_name='Response Variable', value_name='Value')
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(x=factors, y='Value', hue='Response Variable', data=meltedData, legend=False)
        plt.title(f'Box Plot of All Responses Based on {factors}')
        plt.xlabel(factors)
        plt.ylabel('Response Value')

        for line in ax.artists:
            avgLine = line.get_paths()[0]
            ax.plot(avgLine.vertices[:, 0], avgLine.vertices[:, 1], linestyle='--', color='black', linewidth=2)
        plt.show()
    
    else:
        clear()
        print("[x] No data is available. Please load the data first. [x]")

def main():
    data = None
    while True:
        print("1. Read Data")
        print("2. Data Summary")
        print("3. Display Data")
        print("4. One-Way ANOVA")
        print("5. Two-Way ANOVA")
        print("6. Data Plot")
        print("7. Exit")
        choice = input("[>] Enter Choice (1-7): ")
        
        if choice == "1":
            data = read_data()
        elif choice == "2":
            show_data(data)
        elif choice == "3":
            summary_data(data)
        elif choice == "4":
            onewayANOVA(data)
        elif choice == "5":
            twowayANOVA(data)
        elif choice == "6":
            plot_data(data)
        elif choice == "7":
            print("[!] Thank you for using this program [!]")
            break
        else:
            clear()
            print("[x] Invalid selection. [x]")

if __name__ == "__main__":
    main()
