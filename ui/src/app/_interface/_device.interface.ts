import { IMapping } from "./_mapping.interface";
import { IRecord } from "./_record.interface";

export interface IDevice {
  address: number,
  name: string,
  description: string,
  old_name?: string, 
  access_no: number,
  identification: string,
  manufacturer: string,
  medium: string,
  sign: string,
  status: string,
  type: string,
  version: number,
  skip_cycles: number,
  scope: number,
  pre_poll_delay: number,
  post_poll_delay: number,
  poll_repeat: number,
  comm_status: string,
  log_level: number,
  mapping: Array<IMapping>
}
