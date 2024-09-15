
export interface IMapping {
  id: string,
  mbus_type: string,
  mbus_unit: string,
  mbus_value: number,
  instance: number,
  type: string,
  name: string,
  description: string,
  multiplier: number,
  decimal_places: number,
  unit: string,
  type_selection: string[],
  active: boolean
}
