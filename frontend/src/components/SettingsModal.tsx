import React from 'react';
import { X, Save, Shield, HardDrive, DollarSign, Volume2, Search } from 'lucide-react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: {
    userId: string;
    preferredModel: string;
    privacyLevel: string;
    autoPlayAudio: boolean;
    searchEnabled: boolean;
    searchProviders: string;
    monthlyBudgetLimit: number;
  };
  onSave: (newSettings: any) => void;
}

const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose, settings, onSave }) => {
  const [localSettings, setLocalSettings] = React.useState(settings);

  if (!isOpen) return null;

  const handleSave = () => {
    onSave(localSettings);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-70 backdrop-blur-sm p-4">
      <div className="bg-gray-800 border border-gray-700 rounded-xl shadow-2xl w-full max-w-2xl overflow-hidden">
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div className="flex items-center space-x-2">
            <div className="p-2 bg-blue-600 rounded-lg">
              <Shield size={20} className="text-white" />
            </div>
            <h2 className="text-xl font-bold text-white">System Settings</h2>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors">
            <X size={24} />
          </button>
        </div>

        <div className="p-6 overflow-y-auto max-h-[70vh]">
          <div className="space-y-8">
            {/* User Configuration */}
            <section>
              <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">User Profile</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-1">User ID</label>
                  <input
                    type="text"
                    value={localSettings.userId}
                    onChange={(e) => setLocalSettings({ ...localSettings, userId: e.target.value })}
                    className="w-full px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none text-white"
                  />
                </div>
              </div>
            </section>

            {/* AI Model Settings */}
            <section>
              <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4 flex items-center">
                <HardDrive size={16} className="mr-2" /> AI & Hardware
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Preferred Model Tier</label>
                  <select
                    value={localSettings.preferredModel}
                    onChange={(e) => setLocalSettings({ ...localSettings, preferredModel: e.target.value })}
                    className="w-full px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none text-white"
                  >
                    <option value="auto">Automatic (Hardware Detection)</option>
                    <option value="light">Light (Fast, Low Memory)</option>
                    <option value="medium">Medium (Balanced)</option>
                    <option value="heavy">Heavy (High Reasoning, High Memory)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Privacy Level</label>
                  <div className="flex space-x-2">
                    {['low', 'medium', 'high'].map((level) => (
                      <button
                        key={level}
                        onClick={() => setLocalSettings({ ...localSettings, privacyLevel: level })}
                        className={`flex-1 px-4 py-2 rounded-lg border transition-all ${
                          localSettings.privacyLevel === level
                            ? 'bg-blue-600 border-blue-500 text-white shadow-lg shadow-blue-900/20'
                            : 'bg-gray-900 border-gray-700 text-gray-400 hover:border-gray-500'
                        }`}
                      >
                        {level.charAt(0).toUpperCase() + level.slice(1)}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </section>

            {/* Interaction Settings */}
            <section>
              <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4 flex items-center">
                <Volume2 size={16} className="mr-2" /> Interaction
              </h3>
              <div className="flex items-center justify-between p-4 bg-gray-900 rounded-lg border border-gray-700">
                <div>
                  <div className="font-medium text-white">Auto-play Voice Responses</div>
                  <div className="text-xs text-gray-500">Speak generated responses automatically</div>
                </div>
                <button
                  onClick={() => setLocalSettings({ ...localSettings, autoPlayAudio: !localSettings.autoPlayAudio })}
                  className={`w-12 h-6 rounded-full transition-colors relative ${
                    localSettings.autoPlayAudio ? 'bg-blue-600' : 'bg-gray-700'
                  }`}
                >
                  <div className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${
                    localSettings.autoPlayAudio ? 'translate-x-6' : 'translate-x-0'
                  }`} />
                </button>
              </div>
            </section>

            {/* Search Configuration */}
            <section>
              <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4 flex items-center">
                <Search size={16} className="mr-2" /> Search & Retrieval
              </h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-gray-900 rounded-lg border border-gray-700">
                  <div>
                    <div className="font-medium text-white">Enable Web Search</div>
                    <div className="text-xs text-gray-500">Allow agent to search the web for missing info</div>
                  </div>
                  <button
                    onClick={() => setLocalSettings({ ...localSettings, searchEnabled: !localSettings.searchEnabled })}
                    className={`w-12 h-6 rounded-full transition-colors relative ${
                      localSettings.searchEnabled ? 'bg-blue-600' : 'bg-gray-700'
                    }`}
                  >
                    <div className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${
                      localSettings.searchEnabled ? 'translate-x-6' : 'translate-x-0'
                    }`} />
                  </button>
                </div>
                {localSettings.searchEnabled && (
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Search Providers (comma separated)</label>
                    <input
                      type="text"
                      value={localSettings.searchProviders}
                      onChange={(e) => setLocalSettings({ ...localSettings, searchProviders: e.target.value })}
                      placeholder="duckduckgo,bing,google"
                      className="w-full px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none text-white"
                    />
                  </div>
                )}
              </div>
            </section>

            {/* Budget Governance */}
            <section>
              <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4 flex items-center">
                <DollarSign size={16} className="mr-2" /> Budgeting
              </h3>
              <div className="p-4 bg-gray-900 rounded-lg border border-gray-700">
                <label className="block text-sm text-gray-400 mb-1">Monthly Spending Limit (USD)</label>
                <div className="flex items-center space-x-2">
                  <span className="text-gray-500">$</span>
                  <input
                    type="number"
                    value={localSettings.monthlyBudgetLimit}
                    onChange={(e) => setLocalSettings({ ...localSettings, monthlyBudgetLimit: parseFloat(e.target.value) })}
                    className="w-full px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none text-white"
                  />
                </div>
              </div>
            </section>
          </div>
        </div>

        <div className="flex items-center justify-end p-6 border-t border-gray-700 bg-gray-800/50 space-x-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="flex items-center space-x-2 px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors shadow-lg shadow-blue-900/30"
          >
            <Save size={18} />
            <span>Save Changes</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
