import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { IResponse } from '../_interface/_response.interface';
import { IDevice } from '../_interface/_device.interface';
import { Observable } from 'rxjs';
import { ISupportedDevices } from '../_interface/_supported-devices.interface';
import { ITemplate } from '../_interface/_template.interface';
import { IMapping } from '../_interface/_mapping.interface';

@Injectable({
  providedIn: 'root'
})
export class DeviceService {

    constructor() {
    }

    private http = inject(HttpClient)

    getManufacturerList(): Observable<Object>{
      return this.http.get<Object>('/de/assets/manufacturer-list.json');
    }

    getMedia(): Observable<Object>{
      return this.http.get<Object>('/de/assets/media-list.json');
    }

    setDevices(devices: IDevice[]){
      return this.http.post<IResponse>('/api/device/setdevices', devices);
    }

    getDevices(): Observable<IDevice[] | IResponse>{
      return this.http.get<IDevice[] | IResponse>('/api/device/getdevices');
    }

    getDevicesList(): Observable<IDevice[] | IResponse>{
      return this.http.get<IDevice[] | IResponse>('/api/device/getdeviceslist');
    }

    getResourceProfile(scope: number){
      return this.http.post<IDevice|ITemplate>('/api/device/getresourceprofile', scope);
    }

    setResourceProfile(resource: IDevice|ITemplate){
      return this.http.post<IResponse>('/api/device/setresourceprofile', resource);
    }

    setResourceProperty(device: IDevice|ITemplate, property: string){
      return this.http.post<IResponse>('/api/device/setresourceproperty', {'scope': device.scope, 'property': property, 'value': device[property]});
    }

    getSupportedDevices(): Observable<ISupportedDevices | IResponse>{
      return this.http.get<ISupportedDevices | IResponse>('/api/mbus/getsupporteddevices');
    }

    deleteDevices(scopes: number[]){
      return this.http.post<IResponse>('/api/device/deletedevices', scopes);
    }

    setTemplates(templates: IDevice[]){
      return this.http.post<IResponse>('/api/device/settemplates', templates);
    }

    getTemplates(): Observable<ITemplate[] | IResponse>{
      return this.http.post<ITemplate[] | IResponse>('/api/device/gettemplatesbyversion', {manufacturer: '', version: 0 });
    }

    getTemplatesList(): Observable<ITemplate[] | IResponse>{
      return this.http.get<ITemplate[] | IResponse>('/api/device/gettemplateslist');
    }

    getTemplatesByVersion(manufacturer: string, version: number): Observable<ITemplate[] | IResponse>{
      return this.http.post<ITemplate[] | IResponse>('/api/device/gettemplatesbyversion', {manufacturer: manufacturer, version: version });
    }

    editTemplates(templates: ITemplate[]){
      return this.http.post<IResponse>('/api/device/edittemplates', templates);
    }

    deleteTemplate(templates: number[]){
      return this.http.post<IResponse>('/api/device/deletetemplate', templates);
    }

    applyTemplate(templateScope: number, deviceScopes: number[]){
      return this.http.post<IResponse>('/api/device/applytemplate', {template_scope: templateScope, device_scopes: deviceScopes});
    }

  }

