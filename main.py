from etf_grid import SingleETFGrid
if __name__ == "__main__":
    strategy = SingleETFGrid()
    result = strategy.run_backtest(start_date='20240101', end_date='20241231')
    
   
    
     # 正确输出统计结果（修改这里）
    if not result.empty:
        stats = strategy._calculate_stats(result)
        print("\n回测统计:")
        for k, v in stats.items():
            print(f"{k}: {v}")