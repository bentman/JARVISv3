import React, { useState, useRef, useEffect } from 'react';
import { Mic, StopCircle } from 'lucide-react';

interface VoiceRecorderProps {
  onTranscription: (blob: Blob) => void;
  isRecording: boolean;
  setIsRecording: (isRecording: boolean) => void;
}

const VoiceRecorder: React.FC<VoiceRecorderProps> = ({ onTranscription, isRecording, setIsRecording }) => {
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const [audioChunks, setAudioChunks] = useState<Blob[]>([]);

  const handleStartRecording = async () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder.current = new MediaRecorder(stream);
        mediaRecorder.current.ondataavailable = (event) => {
          setAudioChunks((prev) => [...prev, event.data]);
        };
        mediaRecorder.current.start();
        setIsRecording(true);
      } catch (err) {
        console.error("Error accessing microphone:", err);
      }
    }
  };

  const handleStopRecording = () => {
    if (mediaRecorder.current) {
      mediaRecorder.current.stop();
      setIsRecording(false);
    }
  };

  useEffect(() => {
    if (!isRecording && audioChunks.length > 0) {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      console.log("Recorded audio blob:", audioBlob);
      onTranscription(audioBlob);
      setAudioChunks([]);
    }
  }, [isRecording, audioChunks, onTranscription]);

  return (
    <div>
      <button
        onMouseDown={handleStartRecording}
        onMouseUp={handleStopRecording}
        onTouchStart={handleStartRecording}
        onTouchEnd={handleStopRecording}
        className={`p-3 rounded-full transition-colors ${
          isRecording ? 'bg-red-500 text-white' : 'bg-gray-700 hover:bg-gray-600'
        }`}
      >
        {isRecording ? <StopCircle size={24} /> : <Mic size={24} />}
      </button>
    </div>
  );
};

export default VoiceRecorder;
