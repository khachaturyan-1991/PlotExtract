import { IMapping } from "./_mapping.interface";

export interface ITemplate {
  address: number,
  name: string,
  description: string,
  old_name?: string,
  manufacturer: string,
  version: number,
  medium: string,
  sign: string,
  status: string,
  type: string,
  skip_cycles: number,
  scope: number,
  pre_poll_delay: number,
  post_poll_delay: number,
  poll_repeat: number,
  log_level: number,
  mapping: IMapping[],
  newTemplate: Boolean
}

