import { useState, FormEvent } from "react";

import "./Chooser.scss";

export interface ChooserProps {
    options: string[];
    startOption: string;
    onChoose?: (opt: string) => void;
}

export default function Chooser({
    options,
    startOption,
    onChoose = () => {},
}: ChooserProps) {
    const [option, setOption] = useState(startOption);

    function handleChange(e: FormEvent<HTMLDivElement>) {
        const target = e.target as HTMLInputElement;
        onChoose(target.value);
        setOption(target.value);
    }

    return (
        <div className="chooser">
            {options.map((opt) => (
                <div
                    key={opt}
                    className={
                        "chooser__item" +
                        (opt == option ? " chooser__item_active" : "")
                    }
                >
                    <input
                        type="radio"
                        name="text-source"
                        id={"chooser_opt_" + opt}
                        value={opt}
                        checked={opt == option}
                        onChange={handleChange}
                    />
                    <label
                        className="chooser__item__content"
                        htmlFor={"chooser_opt_" + opt}
                    >
                        {opt}
                    </label>
                </div>
            ))}
        </div>
    );
}
