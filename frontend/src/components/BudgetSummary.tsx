import React, { useEffect, useState } from 'react';
import { systemService, BudgetStatus } from '../services/api';

const BudgetSummary: React.FC = () => {
  const [status, setStatus] = useState<BudgetStatus | null>(null);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const data = await systemService.getBudgetStatus();
        setStatus(data);
      } catch (error) {
        console.error('Failed to fetch budget status:', error);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 30000); // 30s update
    return () => clearInterval(interval);
  }, []);

  if (!status) return null;

  return (
    <div className="flex flex-col gap-1 text-xs bg-gray-100 p-2 rounded border border-gray-200">
      <div className="flex justify-between font-bold text-gray-700">
        <span>Monthly Budget</span>
        <span>${status.cloud_spend_usd.toFixed(2)} / ${status.monthly_limit_usd.toFixed(2)}</span>
      </div>
      <div className="w-full bg-gray-300 h-2 rounded-full overflow-hidden">
        <div 
          className={`h-full ${status.remaining_pct < 10 ? 'bg-red-500' : 'bg-blue-500'}`}
          style={{ width: `${100 - status.remaining_pct}%` }}
        ></div>
      </div>
      <div className="flex justify-between text-gray-500 italic">
        <span>Daily: ${status.daily_spending.toFixed(2)}</span>
        <span>{status.remaining_pct.toFixed(1)}% Left</span>
      </div>
    </div>
  );
};

export default BudgetSummary;
