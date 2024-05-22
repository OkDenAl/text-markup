import { Dispatch, useReducer } from "react";

export function useReducerWithMiddleware<T, A>(
    reducer: (state: T, action: A) => T,
    initialState: T,
    middlewares?: ((action: A, dispatch: Dispatch<A>, state: T) => A)[]
): [T, Dispatch<A>] {
    const [state, dispatch] = useReducer(reducer, initialState);

    const dispatchWithMiddlewares = (action: A) => {
        if (middlewares) {
            action = middlewares.reduce((act, middleware) => {
                act = middleware(act, dispatchWithMiddlewares, state);
                if (!act)
                    throw new TypeError("middleware should return action");
                return act;
            }, action);
        }
        dispatch(action);
    };

    return [state, dispatchWithMiddlewares];
}
