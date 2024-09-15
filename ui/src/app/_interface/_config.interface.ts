export interface IConfig {
  language: string,
  port: string,
  auto_assign_port: boolean,
  poll_frequency: number,
  interdevice_delay: number,
  baudrate: number,
  retries: number,
  number_of_addresses: number,
  device_config: string,
  offline_skip_cycles: number,
  template_config: string,
  enable_mbus_comm: boolean,
  reinitialize_config: boolean,
  log_level: number,
  file_log_level: number
}
