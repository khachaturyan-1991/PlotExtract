import { APP_INITIALIZER, Injector, NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';
import {TranslateLoader, TranslateModule, TranslateService} from '@ngx-translate/core';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { NavMenuComponent } from './nav-menu/nav-menu.component';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { appInitializerFactory, createTranslateLoader } from './util/translation.functions';
import { ConfirmationService, MessageService, PrimeNGConfig } from 'primeng/api';
import { ButtonModule } from 'primeng/button';
import { MenubarModule } from 'primeng/menubar';
import { ToastModule } from 'primeng/toast';
import { ChartModule } from 'primeng/chart';
import { InputNumberModule } from 'primeng/inputnumber';
import { ChartViewComponent } from './chart-view/chart-view.component';
import { UploadViewComponent } from './upload-view/upload-view.component';
import { BadgeModule } from 'primeng/badge';
import { DropdownModule } from 'primeng/dropdown';
import { FileUploadModule } from 'primeng/fileupload';
import { ConfigViewComponent } from './config-view/config-view.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { PlotService } from './_service/plot.service';
import { LanguageSelectorComponent } from './language-selector/language-selector.component';

@NgModule({
  declarations: [
    AppComponent,
    NavMenuComponent,
    ChartViewComponent,
    UploadViewComponent,
    ConfigViewComponent,
    LanguageSelectorComponent
  ],
  imports: [
    BrowserModule,
    FormsModule,
    AppRoutingModule,
    ButtonModule,
    MenubarModule,
    ToastModule,
    InputNumberModule,
    HttpClientModule,
    ChartModule,
    BadgeModule,
    DropdownModule,
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
    PlotService,
    {
      provide: APP_INITIALIZER,
      useFactory: appInitializerFactory,
      deps: [TranslateService, Injector],
      multi: true
    }],
  bootstrap: [AppComponent]
})
export class AppModule { }
