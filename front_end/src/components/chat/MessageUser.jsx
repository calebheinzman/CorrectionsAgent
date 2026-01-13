import { Chip } from '../primitives';

export function MessageUser({ message, timestamp, tags = [] }) {
  const formatTime = (date) => {
    if (!date) return '';
    return new Date(date).toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
    });
  };

  return (
    <div className="flex justify-end mb-4">
      <div className="max-w-[80%]">
        {tags.length > 0 && (
          <div className="flex justify-end gap-1 mb-1">
            {tags.map((tag) => (
              <Chip key={tag} variant="accent" className="text-[10px]">
                {tag}
              </Chip>
            ))}
          </div>
        )}
        <div className="group relative">
          <div className="bg-accent text-white px-4 py-3 rounded-2xl rounded-br-md">
            <p className="text-sm whitespace-pre-wrap">{message}</p>
          </div>
          <div className="absolute -bottom-5 right-0 opacity-0 group-hover:opacity-100 transition-opacity">
            <span className="text-xs text-gray-400">{formatTime(timestamp)}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
