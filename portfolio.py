import datetime as dt
import matplotlib.pyplot as plt
import yfinance as yf

class PortfolioVisualizer:
    def __init__(self, tickers, amounts):
        self.tickers = tickers
        self.amounts = amounts
        self.prices = []
        self.total = []

    def download_prices(self, start_date, end_date):
        for ticker in self.tickers:
            df = yf.download(ticker, start_date, end_date)
            price = df[-1:]['Close'][0]
            self.prices.append(price)
            index = self.prices.index(price)
            self.total.append(price * self.amounts[index])

    def plot_portfolio(self):
        fig, ax = plt.subplots(figsize=(16, 8))

        ax.set_facecolor('black')
        ax.figure.set_facecolor('#121212')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        ax.set_title("NEURALNINE PORTFOLIO VISUALIZER", color='#EF6C35', fontsize=20)

        patches, texts, autotexts = ax.pie(self.total, labels=self.tickers, autopct='%1.1f%%', pctdistance=0.8)
        [text.set_color('white') for text in texts]

        my_circle = plt.Circle((0, 0), 0.55, color='black')
        plt.gca().add_artist(my_circle)

        ax.text(-2, 1, 'PORTFOLIO OVERVIEW:', fontsize=14, color="#ffe536", horizontalalignment='center',
                verticalalignment='center')
        ax.text(-2, 0.85, f'Total USD Amount: {sum(self.total):.2f} $', fontsize=12, color="white",
                horizontalalignment='center', verticalalignment='center')
        counter = 0.15
        for ticker in self.tickers:
            ax.text(-2, 0.85 - counter, f'{ticker}: {self.total[self.tickers.index(ticker)]:.2f} $', fontsize=12,
                    color="white", horizontalalignment='center', verticalalignment='center')
            counter += 0.15

        plt.show()


tickers = ['WFC', 'AAPL', 'TSLA', 'GOOG', 'GE']
amounts = [12, 16, 12, 11, 7]
start_date = dt.datetime(2019, 8, 1)
end_date = dt.datetime.now()

portfolio_visualizer = PortfolioVisualizer(tickers, amounts)
portfolio_visualizer.download_prices(start_date, end_date)
portfolio_visualizer.plot_portfolio()
