import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { IConfig } from '../_interface/_config.interface';
import { IResponse } from '../_interface/_response.interface';
import { IPort } from '../_interface/_port.interface';

@Injectable({
  providedIn: 'root'
})
export class ConfigService {
    config$: Observable<IConfig | IResponse>;

    constructor() {
    }

    private http = inject(HttpClient);

    getPorts(): Observable<IPort[] | IResponse> {
      return this.http.get<IPort[] | IResponse>('/api/mbus/getports');
    }

    getConfig(): Observable<IConfig | IResponse> {
      if (!this.config$){
        this.config$ = this.http.get<IConfig | IResponse>('/api/gateway/getconfig');
      }

      return this.config$;
    }

    save($data: IConfig): Observable<IResponse> {
      return this.http.post<IResponse>('/api/gateway/setconfig', $data);
    }
  }

