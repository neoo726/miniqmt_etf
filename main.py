import numpy as np
import pandas as pd
from xtquant import xtdata
import matplotlib.pyplot as plt

# 设置matplotlib配置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 简化配置
CONFIG = {
    "etf": "510300.SH",          # 只交易沪深300ETF
    "initial_capital": 10000,    # 初始本金
    "grid_spacing": 0.02,        # 2%网格间距
    "trade_ratio": 0.3,          # 单次交易30%
    "fee_rate": 0.00005,         # 万0.5手续费
    "slippage": 0.01             # 滑点
}
# 沪深300基准参考
def get_hs300_data(start_date, end_date):
    """获取沪深300指数数据"""
    xtdata.download_history_data(stock_code='000300.SH', period='1d', 
                               start_time=start_date, end_time=end_date)
    data = xtdata.get_market_data(stock_list=['000300.SH'], period='1d',
                                start_time=start_date, end_time=end_date)
    df = pd.DataFrame({
        'date': pd.to_datetime(data['close'].columns.astype(str)),
        'hs300_close': data['close'].loc['000300.SH'].values
    })
    df.set_index('date', inplace=True)
    return df

class SingleETFGrid:
    def __init__(self):
        self.position = 0
        self.cash = CONFIG['initial_capital']
        self.trade_log = []
        self.total_cost = 0  # 新增总成本跟踪
    
    def run_backtest(self, start_date='20220101', end_date='20250131'):
        """回测执行"""
        # 确保数据已下载
        xtdata.download_history_data(
            stock_code=CONFIG['etf'],
            period='1d',
            start_time=start_date,
            end_time=end_date
        )

         # 修改这里 → 调用数据验证方法
        df = self._get_verified_data(start_date, end_date)  # 替换原来的get_market_data调用
       
        # 添加空值检查
        if df is None:
            raise ValueError(f"未能获取{CONFIG['etf']}的历史数据，请检查数据下载是否成功")
            
        for date in df.index:
            daily_data = df.loc[date]
            price = daily_data['close']
            # 生成交易信号（新增这行）
            signal = self._generate_signal(price)  
            # 使用date作为时间戳
            self._execute_trade(price, signal, date)
        
        result = self._generate_report()
        if not result.empty:
            print("回测结果摘要：")
        # 使用合并后的正确字段
        print(result[['strategy_cum', 'hs300_cum']].tail().apply(lambda x: x*100).round(2))
        # 最终状态直接从对象属性获取（关键修改）
        print("\n最终持仓状态:")
        print(f"现金: {self.cash:.2f} 元")
        print(f"持仓: {self.position} 股")
        print(f"组合净值: {self.cash + self.position * price:.2f} 元")  # 使用最后一天的price
        
        return result

    def _generate_signal(self, price):
        """修正后的信号生成"""
        if self.position == 0:
            return 'buy'
            
        # 正确计算平均成本
        avg_cost = self.total_cost / (self.position * 100)  # 每股成本
        current_return = (price - avg_cost) / avg_cost
        
        if current_return <= -CONFIG['grid_spacing']:
            return 'buy'
        elif current_return >= CONFIG['grid_spacing']:
            return 'sell'
        return 'hold'

    def _execute_trade(self, price, signal, timestamp):
         # 确保时间戳格式正确
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
            
        """修正后的交易执行"""
        if signal == 'buy':
            # 计算实际交易量（按100股整数倍）
            trade_amount = self.cash * CONFIG['trade_ratio']
            volume = int(trade_amount // (price * 100)) * 100  # 100股为最小单位
            
            if volume > 0:
                # 计算实际成本（含手续费和滑点）
                adjusted_price = price * (1 + CONFIG['slippage'])
                fee = adjusted_price * volume * CONFIG['fee_rate']
                cost = adjusted_price * volume + fee
                
                self.total_cost += cost
                self.position += volume
                self.cash -= cost
               
                
            
        elif signal == 'sell':
            if self.position > 0:
                # 计算实际卖出量
                volume = int(self.position * CONFIG['trade_ratio'])
                volume = max(volume, 100)  # 至少卖出100股
                volume = min(volume, self.position)
                
                # 计算实际收入（扣除手续费和滑点）
                adjusted_price = price * (1 - CONFIG['slippage']) 
                fee = adjusted_price * volume * CONFIG['fee_rate']
                income = adjusted_price * volume - fee
                
                # 按比例减少总成本
                sell_ratio = volume / self.position
                self.total_cost *= (1 - sell_ratio)
                
                self.position -= volume
                self.cash += income
        # 在交易发生时记录日志
        if signal in ('buy', 'sell'):
            self._log_trade(
                action=signal,
                price=price,
                volume=volume,  # 来自之前的计算
                timestamp=timestamp
            )
        # 记录最后价格（新增）
        self.last_price = price

    def _log_trade(self, action, price, volume, timestamp):
        """增强日志记录"""
        self.trade_log.append({
        'date': timestamp.strftime('%Y-%m-%d'),
        'action': action,
        'price': float(price),  # 转换为Python原生float类型
        'volume': int(volume),
        'cash': float(self.cash),
        'position': int(self.position),
        'value': float(self.cash + self.position * price)
    })
            
    
    def _generate_report(self):
        """生成包含基准对比的报告"""
        df = pd.DataFrame(self.trade_log)
        if df.empty:
            return df
            
        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df['value'] = df['cash'] + df['position'] * df['price']
        
        # 获取沪深300数据
        hs300 = get_hs300_data(df.index[0].strftime('%Y%m%d'), 
                             df.index[-1].strftime('%Y%m%d'))
        
        # 合并数据
        combined = df[['value']].join(hs300, how='inner')
        
        # 转换为百分比收益
        combined['strategy_pct'] = combined['value'].pct_change().fillna(0)
        combined['hs300_pct'] = combined['hs300_close'].pct_change().fillna(0)
        
        # 计算累计收益
        combined['strategy_cum'] = (1 + combined['strategy_pct']).cumprod() - 1
        combined['hs300_cum'] = (1 + combined['hs300_pct']).cumprod() - 1
        
        # 生成统计指标
        stats = self._calculate_stats(combined)
        self._plot_combined(combined, stats)
        
        return combined
    def _calculate_stats(self, df):
        """计算关键统计指标"""
        total_days = len(df)
        years = total_days / 252
        
        # 策略指标
        total_return = df['strategy_cum'].iloc[-1]
        annual_return = (1 + total_return) ** (1/years) - 1
        max_drawdown = (df['strategy_cum'] - df['strategy_cum'].cummax()).min()
        sharpe = np.sqrt(252) * df['strategy_pct'].mean() / df['strategy_pct'].std()
        
        # 沪深300指标
        hs300_total_return = df['hs300_cum'].iloc[-1]
        hs300_annual = (1 + hs300_total_return) ** (1/years) - 1 if years > 0 else 0
        
        # 交易统计
        trades = pd.DataFrame(self.trade_log)
        sell_trades = trades[trades['action'] == 'sell'].copy()
        if not sell_trades.empty:
            sell_trades['prev_price'] = sell_trades['price'].shift(1)
            win_rate = (sell_trades['price'] > sell_trades['prev_price']).mean()
        else:
            win_rate = 0

         # 交易次数统计
        trades = pd.DataFrame(self.trade_log)
        total_trades = len(trades)
        buy_count = (trades['action'] == 'buy').sum()
        sell_count = (trades['action'] == 'sell').sum()

        return {
            '年化收益率': f"{annual_return*100:.1f}%",
            '总收益率': f"{total_return*100:.1f}%",
            '最大回撤': f"{max_drawdown*100:.1f}%",
            '夏普比率': f"{sharpe:.2f}",
            '交易胜率': f"{win_rate*100:.1f}%",
            '总交易次数': total_trades,
            '买入次数': buy_count,
            '卖出次数': sell_count,
            '沪深300年化': f"{hs300_annual*100:.1f}%",  # 新增指标
        }
    
    def _plot_combined(self, df, stats):
        """绘制带交易标记的收益曲线"""
        plt.figure(figsize=(14, 8))
        
        # 绘制主收益曲线
        plt.plot(df.index, df['strategy_cum']*100, 
                label=f'策略收益 ({stats["总收益率"]})', 
                color='#2ca02c', linewidth=2)
        plt.plot(df.index, df['hs300_cum']*100, 
                label=f'沪深300 ({stats["沪深300年化"]})',
                color='#1f77b4', linestyle='--')
        
        # 设置图表元素
        plt.title('策略收益 vs 沪深300')
        plt.ylabel('累计收益率 (%)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # 添加统计信息
        stats_text = "\n".join([
            f"年化收益率: {stats['年化收益率']}",
            f"沪深300年化: {stats['沪深300年化']}",  # 新增行
            f"最大回撤: {stats['最大回撤']}",
            f"夏普比率: {stats['夏普比率']}",
            f"总交易次数: {stats['总交易次数']}",
            f"买入次数: {stats['买入次数']}",
            f"卖出次数: {stats['卖出次数']}",
            f"交易胜率: {stats['交易胜率']}"
        ])
        plt.annotate(stats_text, xy=(0.72, 0.25), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor='white', 
                            edgecolor='gray'))
        
        # 直接显示图表
        plt.show()
   
    def _get_verified_data(self, start_date, end_date):
        """转换数据格式为可处理的时间序列"""
        raw_data = xtdata.get_market_data(
            field_list=['open', 'high', 'low', 'close', 'volume'],
            stock_list=[CONFIG['etf']],
            period='1d',
            start_time= start_date,
            end_time= end_date,
            dividend_type='front',
            fill_data=True
        )
        # pint(raw_data.keys())
        # 转换数据结构
        df = pd.DataFrame({
            'date': raw_data['close'].columns.astype(str),  # 提取日期列
            'open': raw_data['open'].loc[CONFIG['etf']].values,
            'high': raw_data['high'].loc[CONFIG['etf']].values,
            'low': raw_data['low'].loc[CONFIG['etf']].values,
            'close': raw_data['close'].loc[CONFIG['etf']].values,
            'volume': raw_data['volume'].loc[CONFIG['etf']].values
        })
        
        # 转换为时间序列
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)  # 确保时间顺序
        
        return df
# 使用示例
if __name__ == "__main__":
    strategy = SingleETFGrid()
    result = strategy.run_backtest(start_date='20240101', end_date='20241231')
    
   
    
     # 正确输出统计结果（修改这里）
    if not result.empty:
        stats = strategy._calculate_stats(result)
        print("\n回测统计:")
        for k, v in stats.items():
            print(f"{k}: {v}")