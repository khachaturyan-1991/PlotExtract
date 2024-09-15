import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { IProgress } from '../_interface/_progress.interface';
import { IResponse } from '../_interface/_response.interface';
import { Observable } from 'rxjs';


@Injectable({
  providedIn: 'root'
})
export class ScanService {

  constructor() {
  }

  private http = inject(HttpClient)

  getPrimaryProgress(): Observable<IProgress | IResponse> {
    return this.http.get<IProgress | IResponse>('/api/mbus/getprogress');
  }

  getBroadcastProgress(): Observable<IProgress | IResponse> {
    return this.http.get<IProgress | IResponse>('/api/mbus/getbroadcastprogress');
  }

  setScan(status: boolean, fromAddress: number , toAddress: number){
    return this.http.post<IResponse>('/api/mbus/setscan', {"status": status, "from_address": fromAddress, "to_address": toAddress});
  }

  pingSecondary(status: boolean, address: string){
    return this.http.post<IResponse>('/api/mbus/pingsecondary', {"status": status, "address": address});
  }

  broadcastScan(){
    return this.http.get<IResponse>('/api/mbus/broadcast');
  }

  changeAddress(status: boolean, fromAddress: number|string , toAddress: number, migrateSettings: boolean){
    return this.http.post<IResponse>('/api/mbus/changeaddress', {"status": status, "from_address": fromAddress, "to_address": toAddress, "migrate_settings": migrateSettings});
  }
}


