import React from 'react';
import { CheckCircle2, Circle, Loader2 } from 'lucide-react';

interface WorkflowVisualizerProps {
  currentNodeId: string | null;
  completedNodes: string[];
  failedNodes: string[];
}

const nodes = [
  { id: 'router', label: 'Intent Router' },
  { id: 'context_builder', label: 'Context Builder' },
  { id: 'llm_worker', label: 'LLM Processor' },
  { id: 'validator', label: 'Output Validator' },
];

const WorkflowVisualizer: React.FC<WorkflowVisualizerProps> = ({ currentNodeId, completedNodes, failedNodes }) => {
  return (
    <div className="flex items-center space-x-2 py-2 overflow-x-auto no-scrollbar">
      {nodes.map((node, index) => {
        const isCurrent = currentNodeId === node.id;
        const isCompleted = completedNodes.includes(node.id);
        const isFailed = failedNodes.includes(node.id);

        return (
          <React.Fragment key={node.id}>
            <div className={`flex items-center space-x-2 px-3 py-1.5 rounded-full border transition-all duration-300 ${
              isCurrent 
                ? 'bg-blue-600/20 border-blue-500 text-blue-400 shadow-sm shadow-blue-500/20' 
                : isCompleted 
                  ? 'bg-green-600/10 border-green-500/50 text-green-500' 
                  : isFailed 
                    ? 'bg-red-600/10 border-red-500/50 text-red-500' 
                    : 'bg-gray-800 border-gray-700 text-gray-500'
            }`}>
              {isCurrent ? (
                <Loader2 size={14} className="animate-spin" />
              ) : isCompleted ? (
                <CheckCircle2 size={14} />
              ) : (
                <Circle size={14} />
              )}
              <span className="text-xs font-medium whitespace-nowrap">{node.label}</span>
            </div>
            {index < nodes.length - 1 && (
              <div className={`h-px w-4 ${
                isCompleted ? 'bg-green-500/50' : 'bg-gray-700'
              }`} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
};

export default WorkflowVisualizer;
