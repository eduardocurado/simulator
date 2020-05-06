from IPython import display
from ipywidgets import widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import randint


# Get Generic Year
def get_generic_year(working_days_month=None, months=None):
    working_days = [i for i in range(1, working_days_month + 1)]
    month_year = [i for i in range(1, months + 1)]
    df_year = pd.DataFrame()
    for m in month_year:
        df = pd.DataFrame(columns=["working_day"], data=working_days)
        df["month"] = m
        df_year = pd.concat([df_year, df])
        
    return df_year


def get_random_week_value(df, column, week_day=None):
    if week_day is None:
        week_day = randint(1, 5)
    
    df.loc[df["index"] % 5 == (5 - week_day), "week"] = 1
    df["week"] = df["week"].cumsum()
    df["week"].fillna(method="bfill", inplace=True)
    df[f"{column}_per_week"] = df.groupby("week")[column].cumsum()
    df["max_value"] = df.groupby("week")[f"{column}_per_week"].transform("max")
    df.loc[df[f"{column}_per_week"] != df["max_value"], f"{column}_per_week"] = 0
    df.drop(columns=["max_value", "week"], inplace=True)

    return df


def set_sine_to_value(df, col, f=12, amplitude=0.25, pick_values=[]):
    total_value = df[col].sum()
    even_pick = [x for x in pick_values if x % 2 == 0]
    odd_pick = [x for x in pick_values if x % 2 == 1]
    Fs = len(df)
    sample = len(df)
    x = np.arange(sample)
    y_odd =  np.sin(np.pi * f * x / Fs )
    y_even =  -np.sin(np.pi * f * x / Fs )

    df["value_even"] = (1 + y_even * amplitude) * df[col]
    df["value_odd"] = (1 + y_odd * amplitude) * df[col]
    
    df.loc[~df["month"].isin(even_pick), "value_even"] = 0
    df.loc[~df["month"].isin(odd_pick), "value_odd"] = 0
    
    df["value"] = df["value_even"] + df["value_odd"]
    max_value = df.loc[df["month"].isin(pick_values), "value"].sum()
    
    if max_value > total_value:
        df["value"] = df["value"] - (max_value - total_value)/len(df)
    remaining_fat = 0
    if (len(df) - len(pick_values) * 22 > 0):
        remaining_fat = max(0, (total_value - max_value))/(len(df) - len(pick_values) * 22)
        
    df.loc[~df["month"].isin(pick_values), "value"] = remaining_fat
    
    return df["value"]



def split_nfes_values(df, days_in_between, column):
    list_terms = [l for l in days_in_between if l is not None and l > 0]
    num_receivables_income = len(list_terms)

    df["receivable_value"] = df[column]/num_receivables_income
    cols_par = []
    for e, d in enumerate(days_in_between):
        df[f"receivable_split_{e}"] = df["receivable_value"].shift(days_in_between[e])
        cols_par.append(f"receivable_split_{e}")
    df.fillna(0, inplace=True)
    df["total"] = df[cols_par].sum(axis=1)
    
    return df["total"]
    
    

class Cashflow_simulation:
    
    def __init__(self):
        style = {'description_width': 'initial'}
        self.marging_industry_text = widgets.Text(description='Margem:', placeholder="3%", value="3%", style=style)
        self.income_per_year_text = widgets.IntText(description='Fat. Anual (K):', placeholder=15, value=30000, style=style)
        self.num_employees_text = widgets.IntText(description='# Funcionários:', placeholder=120, value=120, style=style)
        self.avg_salary_text = widgets.IntText(description='Salário Médio (K):', placeholder=4, value=4, style=style)
        self.percentage_receivables_text = widgets.Text(description='% Fornecedores:', placeholder="70%", value="70%", style=style)
        self.cash_text = widgets.IntText(description='$ Caixa (K):', placeholder=1500, value=1500, style=style)
        self.working_days_month_text = widgets.IntText(description='Dias Úteis:', placeholder=22, value=22, style=style)
        self.months_text = widgets.IntText(description='Meses:', placeholder=12, value=12, style=style)
        self.avg_lenght_seller_1 = widgets.IntText(description='Parcela 1:', placeholder=22, value=22, style=style)
        self.avg_lenght_seller_2 = widgets.IntText(description='Parcela 2:', placeholder=35, value=35, style=style)
        self.avg_lenght_seller_3 = widgets.IntText(description='Parcela 3:', placeholder=47, value=47, style=style)
        self.avg_lenght_seller_4 = widgets.IntText(description='Parcela 4:', style=style)
        self.avg_lenght_seller_5 = widgets.IntText(description='Parcela 5:', style=style)
        
        self.avg_lenght_buyer_1 = widgets.IntText(description='Parcela 1:', placeholder=22, value=22, style=style)
        self.avg_lenght_buyer_2 = widgets.IntText(description='Parcela 2:', placeholder=35, value=35, style=style)
        self.avg_lenght_buyer_3 = widgets.IntText(description='Parcela 3:', placeholder=47, value=47, style=style)
        self.avg_lenght_buyer_4 = widgets.IntText(description='Parcela 4:', style=style)
        self.avg_lenght_buyer_5 = widgets.IntText(description='Parcela 5:', style=style)
        
        # Tempo de estoque em dias
        
        self.months_check = []
        self.month_boxs = []
        months = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                 "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
        
        for e, m in enumerate(months):
            self.months_check.append(widgets.Checkbox(description=f"{m}", value=False, indent=False))
            if (e + 1) % 3 == 0:
                self.month_box = widgets.VBox(self.months_check)
                self.month_boxs.append(self.month_box)
                self.months_check = []
            
            
        self.month_boxes = widgets.HBox(self.month_boxs)
        self.df = None
        
        self.out = widgets.Output()
        
        
        
        self.box_inputs = widgets.VBox([
            widgets.HTML("<center><b>Informações Empresa:"),
            self.marging_industry_text,
            self.income_per_year_text,
            self.num_employees_text,
           self.avg_salary_text,
            self.percentage_receivables_text,
            self.cash_text,
            self.working_days_month_text,
            self.months_text
                           ])
        
        self.box_nfes_seller = widgets.VBox([
            widgets.HTML("<center><b>Parcelas pagamento Sacados:"),
            self.avg_lenght_seller_1,
            self.avg_lenght_seller_2,
            self.avg_lenght_seller_3,
            self.avg_lenght_seller_4,
            self.avg_lenght_seller_5,
        ])
        
        self.box_nfes_buyer = widgets.VBox([
            widgets.HTML("<center><b>Parcelas pagamento Fornecedores:"),
            self.avg_lenght_buyer_1,
            self.avg_lenght_buyer_2,
            self.avg_lenght_buyer_3,
            self.avg_lenght_buyer_4,
            self.avg_lenght_buyer_5,
        ])
        
        self.box = widgets.HBox([
            self.box_inputs, widgets.HBox([self.box_nfes_seller, self.box_nfes_buyer])
        ])
        
        
        button = widgets.Button(description="Click Me!")
        display.display(self.box, widgets.VBox([widgets.HTML("<b>Meses com maior Faturamento:"), self.month_boxes]), button, self.out)

        def on_button_clicked(b):
            with self.out:
                self.on_button_clicked()

        button.on_click(on_button_clicked)
        
        
        

    def on_button_clicked(self):
        with self.out:
            self.out.clear_output()
            
            pick_values = []
            for e, x in enumerate(self.month_boxs):
                for i, y in enumerate(x.children):
                    if y.value is True:
                        pick_values.append(((e) * 3) + (i + 1))
            
            if pick_values == []:
                pick_values = list(range(1, 13))
            
        
            # Get Initial Input Values

            marging_industry = float(self.marging_industry_text.value.replace("%", "").strip())/100
            income_per_year = self.income_per_year_text.value
            num_employees = self.num_employees_text.value
            avg_salary = self.avg_salary_text.value # $ Annual total package / 12 months

            percentage_receivables = float(self.percentage_receivables_text.value.replace("%", "").strip())/100
            cash = self.cash_text.value

            working_days_month = self.working_days_month_text.value
            months = self.months_text.value
            

            expense_salary = num_employees * avg_salary * 1.3 # salary factor
            annual_expense_salary = expense_salary * 12
            cost_per_year = (income_per_year/(1 + marging_industry))
            mean_cost_per_month = cost_per_year/12
            annual_cost_receivables = (cost_per_year - annual_expense_salary) * percentage_receivables 
            annual_other_expenses = (cost_per_year - annual_expense_salary) * (1 - percentage_receivables)

            df_year = get_generic_year(working_days_month, months)

            df_year["receivables_per_day"] = annual_cost_receivables/12/working_days_month
            df_year["other_expenses_per_day"] = annual_other_expenses/12/working_days_month
            df_year["salary"] = 0

            working_day_salary = working_days_month
            df_year.loc[df_year["working_day"] == working_days_month, "salary"] = expense_salary
            df_year.reset_index(inplace=True)
            df_year["index"] += 1
            
            df_year.reset_index(inplace=True)
            print(f"Monthly Company Size: R$ {income_per_year/12} K\nCost Per Month: R$ {round(mean_cost_per_month, 2)} K\nSalary: R$ {expense_salary} K (Day 1)\nCaixa: R$ {cash} K")
            
#             df_year.loc[df_year["month"] == 1, "other_expenses_per_day"] = 0
            mean_daily_income = income_per_year/12/working_days_month
            df_year["income_per_day"] = mean_daily_income
            
            df_year["receivables_per_day_sine_per_week"] = set_sine_to_value(df_year.copy(), "receivables_per_day", f=months, pick_values= pick_values)
            df_year["income_per_day_sine_per_week"] = set_sine_to_value(df_year.copy(), "income_per_day", f=months, pick_values=pick_values)
            df_year["other_expenses_per_day_sine_per_week"] = set_sine_to_value(df_year.copy(), "other_expenses_per_day", f=0)
            
#             df_year = get_random_week_value(df_year, "receivables_per_day_sine", 5)
#             df_year = get_random_week_value(df_year, "income_per_day_sine")
#             df_year = get_random_week_value(df_year, "other_expenses_per_day_sine")
            
            # How receivables are paid from buyers
            days_in_between_seller = [self.avg_lenght_seller_1.value,
                               self.avg_lenght_seller_2.value,
                               self.avg_lenght_seller_3.value,
                               self.avg_lenght_seller_4.value,
                               self.avg_lenght_seller_5.value]
            days_in_between_buyer = [self.avg_lenght_buyer_1.value,
                               self.avg_lenght_buyer_2.value,
                               self.avg_lenght_buyer_3.value,
                               self.avg_lenght_buyer_4.value,
                               self.avg_lenght_buyer_5.value]
            
            # How receivables are paid for sellers
            df_year["receivable_income"] = split_nfes_values(df_year.copy(), days_in_between=days_in_between_seller, column="income_per_day_sine_per_week")
            df_year["receivable_cost"] = split_nfes_values(df_year.copy(), days_in_between=days_in_between_buyer, column="receivables_per_day_sine_per_week")

            df_year["cost_per_day_sine"] = df_year["other_expenses_per_day_sine_per_week"] + df_year["receivable_cost"] + df_year["salary"]
            df_year["cost_book_per_day_sine"] = df_year["other_expenses_per_day_sine_per_week"] + df_year["receivables_per_day_sine_per_week"] + df_year["salary"]

            df_year["cash"] = 0
            df_year.loc[0, "cash"] = cash

            df_year["cost_cumsum"] = (df_year["cost_per_day_sine"]).cumsum()
            df_year["income_cumsum"] = (df_year["receivable_income"] + df_year["cash"]).cumsum()

            df_year["cost_book_cumsum"] = df_year["cost_book_per_day_sine"].cumsum()
            df_year["income_book_cumsum"] = (df_year["income_per_day_sine_per_week"] + df_year["cash"]).cumsum()
            df_cashflow = df_year.loc[:, ["working_day", "month", "cost_cumsum", "income_cumsum", "cost_book_cumsum", "income_book_cumsum",
                                         "income_per_day_sine_per_week", "cost_book_per_day_sine"]]
            df_cashflow.reset_index(inplace=True)
    
            df_cashflow["net"] = df_cashflow["income_cumsum"] - df_cashflow["cost_cumsum"]
            df_cashflow["net_book"] = df_cashflow["income_book_cumsum"] - df_cashflow["cost_book_cumsum"]

            NUM_ROWS = 2
            IMGs_IN_ROW = 2
            f, ax = plt.subplots(NUM_ROWS, IMGs_IN_ROW, figsize=(16,8))
            
            ax[0][0].bar(df_cashflow["index"], df_cashflow["net"])
            ax[0][1].bar(df_cashflow["index"], df_cashflow["net_book"])
            ax[1][0].plot(df_cashflow["index"], df_cashflow["income_per_day_sine_per_week"])
            ax[1][1].plot(df_cashflow["index"], df_cashflow["cost_book_per_day_sine"])

            ax[0][0].set_title('Available Cash Per Day - Cashflow')
            ax[0][1].set_title('Available Cash Per Day - Book Value')
            ax[1][0].set_title('Revenue')
            ax[1][1].set_title('Cost')

            title = 'Cashflow Comparison: Real x Book Value'
            f.suptitle(title, fontsize=16)
#             plt.gca().set_ylim(bottom=0)
            plt.show()
            df_year.to_csv("annual_cashflow_raw.csv", sep=";", decimal=',', index=False)
            df_cashflow.to_csv("annual_cashflow.csv", sep=";", decimal=',', index=False)
            
            display.display(df_year[["income_per_day_sine_per_week", "cost_book_per_day_sine"]].sum(),
                            df_year["income_per_day_sine_per_week"].sum()/df_year["cost_book_per_day_sine"].sum() - 1)
            return df_year