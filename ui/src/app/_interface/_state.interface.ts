import { IError } from "./_error.interface";

export interface IState {
  initialized: boolean,
  mode: string,
  error: string,
  error_types?: IError
}
