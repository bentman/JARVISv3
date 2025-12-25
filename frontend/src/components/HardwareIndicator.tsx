import React, { useEffect, useState } from 'react';
import { systemService, HardwareStatus } from '../services/api';

const HardwareIndicator: React.FC = () => {
  const [status, setStatus] = useState<HardwareStatus | null>(null);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const data = await systemService.getHardwareStatus();
        setStatus(data);
      } catch (error) {
        console.error('Failed to fetch hardware status:', error);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  if (!status) return <div className="text-xs text-gray-500">Hardware: Loading...</div>;

  return (
    <div className="flex gap-4 text-xs font-mono bg-gray-900 text-green-400 p-2 rounded shadow-inner">
      <div>
        CPU: <span className={status.cpu_usage > 80 ? 'text-red-500' : ''}>{status.cpu_usage.toFixed(1)}%</span>
      </div>
      <div>
        GPU: <span className={status.gpu_usage > 80 ? 'text-red-500' : ''}>{status.gpu_usage.toFixed(1)}%</span>
      </div>
      <div>
        MEM: {status.memory_available_gb.toFixed(1)}GB FREE
      </div>
      <div className="text-gray-500">
        [{status.available_tiers.join('|')}]
      </div>
    </div>
  );
};

export default HardwareIndicator;
