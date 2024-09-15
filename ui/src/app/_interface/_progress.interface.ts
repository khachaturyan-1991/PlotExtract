export interface IProgress {
  status: boolean,
  start_address?: number,
  end_address?: number,
  current_address?: string,
  progress: number,
  result: boolean,
  found: {primary_address: number, secondary_address: string}[]
}
