import { APP_INITIALIZER, Injector, NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import {TranslateLoader, TranslateModule, TranslateService} from '@ngx-translate/core';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { NavMenuComponent } from './nav-menu/nav-menu.component';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { appInitializerFactory, createTranslateLoader } from './util/translation.functions';
import { ConfirmationService, MessageService, PrimeNGConfig } from 'primeng/api';
import { StateService } from './_service/state.service';
import { ButtonModule } from 'primeng/button';
import { MenubarModule } from 'primeng/menubar';
import { ToastModule } from 'primeng/toast';
import { ChartModule } from 'primeng/chart';
import { ChartViewComponent } from './chart-view/chart-view.component';
import { UploadViewComponent } from './upload-view/upload-view.component';
import { BadgeModule } from 'primeng/badge';
import { FileUploadModule } from 'primeng/fileupload';
import { ConfigViewComponent } from './config-view/config-view.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

@NgModule({
  declarations: [
    AppComponent,
    NavMenuComponent,
    ChartViewComponent,
    UploadViewComponent,
    ConfigViewComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    ButtonModule,
    MenubarModule,
    ToastModule,
    HttpClientModule,
    ChartModule,
    BadgeModule,
    FileUploadModule,
    BrowserAnimationsModule,
    TranslateModule.forRoot(
      {
        loader: {
            provide: TranslateLoader,
            useFactory: createTranslateLoader,
            deps: [HttpClient]
          }
      }
    )
  ],
  providers: [
    MessageService,
    ConfirmationService,
    PrimeNGConfig,
    {
      provide: APP_INITIALIZER,
      useFactory: appInitializerFactory,
      deps: [TranslateService, Injector, StateService],
      multi: true
    }],
  bootstrap: [AppComponent]
})
export class AppModule { }
