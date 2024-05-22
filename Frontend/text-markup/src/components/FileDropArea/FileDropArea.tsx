import cn from "classnames";
import { DragEvent, ReactNode } from "react";
import { toast } from "react-toastify";

import "./FileDropArea.scss";

export interface FileDropAreaProps {
    className?: string;
    children: ReactNode;
    onFile?: (file: File) => void;
}

export default function FileDropArea({
    className,
    children,
    onFile = () => {},
}: FileDropAreaProps) {
    function handleFileDrop(e: DragEvent<HTMLDivElement>) {
        e.preventDefault();
        e.currentTarget.classList.remove("file-drop-area_file-over");
        if (e.dataTransfer.files.length > 1) {
            toast.error("Выберите только один файл");
            return;
        }
        onFile(e.dataTransfer.files[0]);
    }

    function handleFileDragOver(e: DragEvent<HTMLDivElement>) {
        e.preventDefault();
        e.currentTarget.classList.add("file-drop-area_file-over");
    }

    function handleFileDragLeave(e: DragEvent<HTMLDivElement>) {
        e.preventDefault();
        e.currentTarget.classList.remove("file-drop-area_file-over");
    }

    return (
        <div
            className={cn("file-drop-area", className)}
            onDrop={handleFileDrop}
            onDragOver={handleFileDragOver}
            onDragLeave={handleFileDragLeave}
        >
            {children}
        </div>
    );
}
