import { useRef, useState } from "react";

export default function UploadZone({ file, onFileSelected, onClear }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = (fileList) => {
    const picked = fileList[0];
    if (!picked) return;
    const isAudio = picked.name.endsWith(".mp3") || picked.name.endsWith(".wav");
    if (!isAudio) return;
    onFileSelected(picked);
  };

  if (file) {
    const audioUrl = URL.createObjectURL(file);
    return (
      <div>
        <div className="file-row">
          <div>
            <div className="fname">{file.name}</div>
            <div className="fsub">{(file.size / 1024 / 1024).toFixed(2)} MB</div>
          </div>
          <button className="clear" onClick={onClear}>Remove</button>
        </div>
        <audio controls src={audioUrl} />
      </div>
    );
  }

  return (
    <div
      className={`dropzone ${dragOver ? "drag-over" : ""}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        handleFiles(e.dataTransfer.files);
      }}
    >
      <div className="dropzone-title">Drop an audio file here, or click to browse</div>
      <div className="dropzone-sub">MP3 or WAV</div>
      <input
        ref={inputRef}
        type="file"
        accept=".mp3,.wav"
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  );
}
