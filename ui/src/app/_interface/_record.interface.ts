export interface IRecord {
  unit: string,
  value: number,
  id: string,
  bacnet_object: string,
  bacnet_instance: string,
  bacnet_name: string,
  bacnet_description: string,
  bacnet_unit: string,
  multiplier: number,
  active: boolean,
  type_selection: string[]
}
