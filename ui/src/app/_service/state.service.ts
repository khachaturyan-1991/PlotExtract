import { HttpClient } from '@angular/common/http';
import { Injectable, WritableSignal, inject, signal } from '@angular/core';
import { Observable, tap } from 'rxjs';
import { IState } from '../_interface/_state.interface';
import { IResponse } from '../_interface/_response.interface';
import { Router } from '@angular/router';
import { IError } from '../_interface/_error.interface';

@Injectable({
  providedIn: 'root'
})
export class StateService {
  private router = inject(Router);
  private http = inject(HttpClient);
  state: IState;
  mode: string;
  errorTypes: WritableSignal<IError> = signal(undefined);

  hasAccess(): Observable<boolean> {
    return new Observable<boolean>(observer => {
      this.getState().subscribe(
        state => {
          this.state = state;
          this.errorTypes.set(state.error_types);
          if (!state.initialized) {
            observer.next(false);
            this.router.navigate(['/start-up']); // Return false when access is denied
          } else if (state.error){
            observer.next(false);
            this.router.navigate(['/error']);
          }
          else {
            observer.next(true); // Return true when access is granted
          }
          observer.complete();
        },
        error => {
          // Handle error if necessary
          observer.error(error);
        }
      );
    });
  }

  setMode(mode: string): Observable<IResponse> {
    return this.http.post<IResponse>('/api/gateway/setmode', {'mode': mode});
  }

  getState(): Observable<IState> {
    return this.http.get<IState>('/api/gateway/getstatus').pipe(tap({
      next: val => {
        this.mode = val.mode;
      }}
      ));
  }

  getMode(): string {
    if(!this.mode){
      this.getState().subscribe();
    }
    return this.mode;
  }

  getVersion(): Observable<{version: string}> {
    return this.http.get<{version: string}>('/api/gateway/getversion');
  }

  getLanguage(): Observable<{language: string}> {
    return this.http.get<{language: string}>('/api/gateway/getlanguage');
  }
}
