import { useContext } from "react";
import { MarkupContext, MarkupDispatchContext } from "./MarkupProvider";

export function useMarkup() {
    return useContext(MarkupContext)!;
}

export function useMarkupDispatch() {
    return useContext(MarkupDispatchContext)!;
}
